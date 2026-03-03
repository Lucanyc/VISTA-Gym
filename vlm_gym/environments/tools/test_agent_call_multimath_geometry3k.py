#!/usr/bin/env python3
"""
Test MultiMath Server on Geometry3K using Agent-generated tool calls
Fixed version: separate tool call generation and answer generation
"""
#这个是正确的设计agent通过调用工具指令来进行的代码
import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import argparse
import re
import requests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add VLM-Gym to path
sys.path.append('/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista')

from vlm_gym.environments import VisionQAEnv
from vlm_gym.agents.vlm_agent_with_tools import VLMAgentWithTools


class SimpleGeometryTask:
    """Simple task wrapper for Geometry problems"""
    
    def __init__(self, task_id, task_data):
        self.task_id = task_id
        self.task_data = task_data
        self.current_step = 0
        self.max_steps = 10
        
    def setup(self):
        return f"Solve geometry problem: {self.task_data['question']}", {
            "task_type": "geometry",
            "task_id": self.task_id
        }
    
    def get_observation(self):
        return {
            "image_path": self.task_data.get("image_path"),
            "question": self.task_data["question"],
            "task_id": self.task_id,
            "multimath_server_enabled": True
        }
    
    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self.max_steps
        return {}, 0, done, False, {"message": "Step completed"}
    
    def teardown(self):
        pass
    
    def reset(self):
        self.current_step = 0
        return self.get_observation(), {}


class AgentMultiMathTester:
    """Test MultiMath Server using Agent-generated tool calls"""
    
    def __init__(self, 
                 data_path: str,
                 model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 mm_api_url: str = "http://localhost:8001",
                 mm_timeout: int = 60,
                 debug: bool = False):
        """Initialize tester with Agent"""
        
        self.data_path = data_path
        self.debug = debug
        self.model_name = model_name
        
        # MultiMath configuration
        self.mm_config = {
            "api_url": mm_api_url,
            "timeout": mm_timeout
        }
        
        # Check MultiMath Server
        self._check_multimath_server()
        
        # Load dataset
        logger.info(f"Loading dataset from {data_path}")
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        logger.info(f"Loaded {len(self.data)} problems")
        
        self.dataset_dir = Path(data_path).parent
        
        # Get data directory for environment
        data_dir = os.path.dirname(data_path)
        if not data_dir:
            data_dir = "/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/geometry3k"
        
        # Initialize environment
        logger.info("Initializing VisionQA environment with MultiMath...")
        try:
            self.env = VisionQAEnv(
                dataset_path=data_dir,
                max_steps=10,
                enable_actions=False,
                enable_multimath_server=True,
                multimath_server_config=self.mm_config,
                enable_grounding_dino=False,
                enable_deepeyes_tools=False,
                enable_chartmoe=False
            )
            logger.info("✓ Environment initialized with MultiMath Server")
            
            # Verify MultiMath tool is available
            available_tools = self.env.get_available_tools()
            if 'multimath_server' in available_tools:
                logger.info("✓ MultiMath Server tool is available")
            else:
                logger.error("❌ MultiMath Server tool not found!")
                raise RuntimeError("MultiMath Server tool not available")
                
        except Exception as e:
            logger.error(f"Failed to initialize environment: {e}")
            raise
        
        # Initialize Agent
        logger.info(f"Initializing Agent with model: {model_name}")
        
        agent_config = {
            "model_type": "HuggingFace",
            "model_name": model_name,
            "max_new_tokens": 512,
            "temperature": 0.1,
            "device_map": "auto",
            "torch_dtype": "bfloat16",
            "trust_remote_code": True,
            "enable_tools": True,
            "enable_multimath_server": True,
            "max_tool_calls": 3,
            "debug": debug
        }
        
        try:
            self.agent = VLMAgentWithTools(agent_config)
            logger.info("✓ Agent initialized")
            
            if hasattr(self.env, 'tool_manager') and 'multimath_server' in self.env.tool_manager:
                self.agent.multimath_server_tool = self.env.tool_manager['multimath_server']
                logger.info("✓ MultiMath tool instance set in agent")
                
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise
        
        # Results storage
        self.results = []
        self.stats = {
            "total": 0,
            "correct": 0,
            "tool_calls_generated": 0,
            "tool_calls_successful": 0,
            "time_total": 0
        }
    
    def _check_multimath_server(self):
        """Check if MultiMath Server is running"""
        try:
            response = requests.get(f"{self.mm_config['api_url']}/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                logger.info(f"✅ MultiMath Server is running")
                logger.info(f"   - Status: {health.get('status', 'unknown')}")
                logger.info(f"   - Model loaded: {health.get('model_loaded', False)}")
        except Exception as e:
            logger.error(f"❌ Cannot connect to MultiMath Server: {e}")
            raise
    
    def test_single_problem(self, problem_idx: int):
        """Test a single problem using Agent (two-stage process)"""
        if problem_idx >= len(self.data):
            logger.error(f"Problem index {problem_idx} out of range")
            return None
        
        problem = self.data[problem_idx]
        
        print("\n" + "="*70)
        print("TESTING WITH AGENT-GENERATED TOOL CALLS (TWO-STAGE)")
        print("="*70)
        print(f"Problem Index: {problem_idx}")
        print(f"Problem ID: {problem['id']}")
        print(f"Question: {problem['question']}")
        print(f"Ground Truth: {problem['answer']}")
        
        # Get image path
        image_path = problem.get("image_path")
        if not image_path:
            print("❌ No image path")
            return None
        
        if not Path(image_path).is_absolute():
            image_path = self.dataset_dir / image_path
        else:
            image_path = Path(image_path)
        
        if not image_path.exists():
            print(f"❌ Image not found: {image_path}")
            return None
        
        print(f"Image: {image_path}")
        print("="*70)
        
        # Create task wrapper
        task_wrapper = SimpleGeometryTask(problem['id'], {
            'question': problem['question'],
            'image_path': str(image_path),
            'answer': problem['answer']
        })
        
        # Set task entrypoint
        self.env.task_entrypoint = lambda task_id=None, **kwargs: task_wrapper
        
        # Reset environment
        print("\n[Environment] Resetting...")
        obs, info = self.env.reset(task_id=problem['id'])
        print("✓ Environment reset completed")
        
        start_time = time.time()
        
        # ==========================================
        # STAGE 1: Generate Tool Call
        # ==========================================
        print("\n" + "="*50)
        print("STAGE 1: GENERATE TOOL CALL")
        print("="*50)
        
        # Create the exact tool call format
        tool_call_json = {
            "tool": "multimath_server",
            "parameters": {
                "task": "solve",
                "question": problem['question'],
                "image": str(image_path),
                "problem_type": "geometry",
                "output_format": "with_steps"
            }
        }
        
        # Prepare task observation for tool call generation
        task_obs_tool_call = {
            "image_path": str(image_path),
            "question": problem['question'],
            "task_id": problem['id'],
            "system_message": f"""You are solving a geometry problem. Your task is to generate a tool call to MultiMath Server.

Generate EXACTLY this tool call (copy it exactly):

<tool_call>
{json.dumps(tool_call_json, ensure_ascii=False, indent=2)}
</tool_call>

IMPORTANT: 
- Generate ONLY the tool call above
- Do NOT provide any answer or explanation
- Just copy the exact tool call format shown above""",
            "output_format_instruction": "Generate only the <tool_call> block shown above.",
            "multimath_server_enabled": True,
            "force_tool_use": True
        }
        
        # Generate tool call
        print("[Agent] Generating tool call...")
        agent_output_tool_call, _ = self.agent.act(task_obs_tool_call)
        
        if self.debug:
            print(f"\n[DEBUG] Agent output (tool call):")
            print(agent_output_tool_call[:500])
        
        # Parse and execute tool call
        tool_call_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        match = re.search(tool_call_pattern, str(agent_output_tool_call), re.DOTALL)
        
        tool_result = None
        
        if match:
            self.stats["tool_calls_generated"] += 1
            print(f"✓ Agent generated tool call")
            
            try:
                # Parse the tool call
                tool_call_str = match.group(1)
                generated_tool_call = json.loads(tool_call_str)
                
                # Verify parameters
                params = generated_tool_call.get('parameters', {})
                required_params = ['task', 'question', 'image', 'problem_type', 'output_format']
                missing_params = [p for p in required_params if p not in params]
                
                if missing_params:
                    print(f"⚠️ Missing parameters: {missing_params}")
                else:
                    print(f"✓ All required parameters present")
                
                # Execute tool call through environment
                print("\n[Executing] MultiMath Server tool call...")
                obs2, reward, done, truncated, info2 = self.env.step(str(agent_output_tool_call))
                
                # Extract tool result
                action_result = info2.get('action_result', {})
                
                if action_result.get('type') == 'tool_result' and action_result.get('tool') == 'multimath_server':
                    tool_result = action_result.get('result', {})
                    
                    if tool_result.get('success'):
                        self.stats["tool_calls_successful"] += 1
                        print(f"✓ MultiMath execution successful")
                        print(f"  - Answer from MultiMath: {tool_result.get('answer')}")
                        print(f"  - Method: {tool_result.get('method')}")
                        
                        if tool_result.get('steps'):
                            print(f"  - Steps: {len(tool_result['steps'])} steps provided")
                    else:
                        print(f"❌ MultiMath failed: {tool_result.get('error')}")
                else:
                    print(f"❌ Tool execution failed")
                    
            except Exception as e:
                print(f"❌ Error executing tool: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()
        else:
            print("❌ Agent did not generate a valid tool call")
        
        # ==========================================
        # STAGE 2: Generate Answer Based on Tool Result
        # ==========================================
        final_answer = None
        
        if tool_result and tool_result.get('success'):
            print("\n" + "="*50)
            print("STAGE 2: GENERATE ANSWER FROM TOOL RESULT")
            print("="*50)
            
            # Format steps for display
            steps_text = ""
            if tool_result.get('steps'):
                for i, step in enumerate(tool_result['steps'][:3], 1):
                    steps_text += f"Step {i}: {step}\n"
                if len(tool_result['steps']) > 3:
                    steps_text += f"... ({len(tool_result['steps']) - 3} more steps)\n"
            
            # Prepare observation for answer generation
            # DO NOT put the answer directly in the prompt
            task_obs_answer = {
                "image_path": str(image_path),
                "question": problem['question'],
                "task_id": problem['id'],
                "system_message": f"""MultiMath Server has solved the geometry problem.

Original Question: {problem['question']}

MultiMath Calculation Result:
- Method: {tool_result.get('method', 'MultiMath-7B')}
- Calculation successful: {tool_result.get('success', False)}
- Calculated answer: {tool_result.get('answer', 'N/A')}

Calculation Steps:
{steps_text if steps_text else 'No detailed steps provided'}

Your task:
Based on the MultiMath calculation above, provide the final numerical answer to the geometry problem.

Output format:
<answer>[numerical answer only]</answer>

Example: If MultiMath calculated 15, output: <answer>15</answer>""",
                "output_format_instruction": "Output only: <answer>[number]</answer>",
                "multimath_server_enabled": False,  # Disable tools for answer generation
                "force_tool_use": False,
                "tool_feedback": {
                    "tool": "multimath_server",
                    "success": True,
                    "answer": tool_result.get('answer'),
                    "method": tool_result.get('method')
                }
            }
            
            print("[Agent] Generating final answer based on tool result...")
            agent_output_answer, _ = self.agent.act(task_obs_answer)
            
            if self.debug:
                print(f"\n[DEBUG] Agent output (answer):")
                print(agent_output_answer[:500])
            
            # Extract answer
            answer_match = re.search(r'<answer>\s*([^<]+?)\s*</answer>', 
                                    str(agent_output_answer), re.IGNORECASE)
            
            if answer_match:
                final_answer = answer_match.group(1).strip()
                print(f"✓ Agent provided answer: {final_answer}")
            else:
                # Fallback: use tool result directly
                final_answer = str(tool_result.get('answer', ''))
                print(f"⚠️ Using MultiMath answer directly: {final_answer}")
        else:
            print("\n❌ Cannot generate answer: tool execution failed")
        
        elapsed_time = time.time() - start_time
        
        # Check correctness
        is_correct = False
        if final_answer:
            is_correct = self.check_answer(final_answer, problem['answer'])
        
        # Print results
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        print(f"Agent's Answer: {final_answer}")
        print(f"Ground Truth: {problem['answer']}")
        print(f"Correct: {'✅' if is_correct else '❌'}")
        print(f"Total Time: {elapsed_time:.2f}s")
        print(f"Tool call generated: {match is not None}")
        print(f"Tool call successful: {tool_result and tool_result.get('success', False)}")
        print("="*70)
        
        return {
            'problem_id': problem['id'],
            'question': problem['question'],
            'ground_truth': problem['answer'],
            'prediction': final_answer,
            'correct': is_correct,
            'time': elapsed_time,
            'tool_call_generated': match is not None,
            'tool_call_successful': tool_result and tool_result.get('success', False)
        }
    
    def check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth"""
        if predicted is None or ground_truth is None:
            return False
        
        # Clean answers
        pred_clean = str(predicted).strip().lower()
        gt_clean = str(ground_truth).strip().lower()
        
        # Remove common suffixes
        for suffix in [' degrees', ' degree', '°', ' units', ' cm', ' m', ' km']:
            pred_clean = pred_clean.replace(suffix, '')
            gt_clean = gt_clean.replace(suffix, '')
        
        # Try exact match
        if pred_clean == gt_clean:
            return True
        
        # Try numeric comparison
        try:
            pred_nums = re.findall(r'-?\d*\.?\d+', pred_clean)
            gt_nums = re.findall(r'-?\d*\.?\d+', gt_clean)
            
            if pred_nums and gt_nums:
                pred_num = float(pred_nums[0])
                gt_num = float(gt_nums[0])
                
                # Check with tolerance
                abs_tol = 0.01
                rel_tol = 1e-4
                
                return (abs(pred_num - gt_num) < abs_tol or 
                       abs(pred_num - gt_num) / max(abs(pred_num), abs(gt_num), 1) < rel_tol)
        except:
            pass
        
        return False
    
    def run_batch_test(self, start_idx: int = 0, num_problems: int = 10):
        """Run batch test on multiple problems"""
        print(f"\n{'='*70}")
        print(f"BATCH TEST: Testing {num_problems} problems starting from index {start_idx}")
        print(f"{'='*70}")
        
        end_idx = min(start_idx + num_problems, len(self.data))
        
        for idx in range(start_idx, end_idx):
            result = self.test_single_problem(idx)
            if result:
                self.results.append(result)
                self.stats["total"] += 1
                if result['correct']:
                    self.stats["correct"] += 1
                self.stats["time_total"] += result['time']
        
        # Print summary
        if self.stats["total"] > 0:
            print(f"\n{'='*70}")
            print("BATCH TEST SUMMARY")
            print(f"{'='*70}")
            print(f"Total problems: {self.stats['total']}")
            print(f"Correct: {self.stats['correct']}")
            print(f"Accuracy: {self.stats['correct']/self.stats['total']*100:.1f}%")
            print(f"Tool calls generated: {self.stats['tool_calls_generated']}")
            print(f"Tool calls successful: {self.stats['tool_calls_successful']}")
            print(f"Average time: {self.stats['time_total']/self.stats['total']:.2f}s")
            print(f"{'='*70}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test MultiMath with Agent-generated tool calls")
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/geometry3k/converted_train/geometry3k_train_vlmgym.json",
        help="Path to geometry3k JSON file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="VLM model name for agent"
    )
    parser.add_argument(
        "--mm_api_url",
        type=str,
        default="http://localhost:8001",
        help="MultiMath server API URL"
    )
    parser.add_argument(
        "--single_problem",
        type=int,
        default=None,
        help="Test a single problem by index"
    )
    parser.add_argument(
        "--batch_test",
        type=int,
        nargs=2,
        metavar=('START', 'NUM'),
        help="Test a batch of problems (start_index, num_problems)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Create tester
    tester = AgentMultiMathTester(
        data_path=args.data_path,
        model_name=args.model,
        mm_api_url=args.mm_api_url,
        debug=args.debug
    )
    
    # Run test
    if args.single_problem is not None:
        tester.test_single_problem(args.single_problem)
    elif args.batch_test:
        start_idx, num_problems = args.batch_test
        tester.run_batch_test(start_idx, num_problems)
    else:
        # Default: test first problem
        tester.test_single_problem(0)


if __name__ == "__main__":
    main()