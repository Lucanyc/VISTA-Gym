#!/usr/bin/env python3
"""
Test GeometrySolver (DF+MM combined tool) on Geometry3K dataset
Modified version: CDL as auxiliary hints with detailed debugging
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import argparse
from PIL import Image
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

# Import tools
from vlm_gym.environments.tools.geometry_tools.diagram_formalizer import DiagramFormalizerTool
from vlm_gym.environments.tools.base import ToolBase


class GeometrySolverTool(ToolBase):
    """几何问题组合求解器 - CDL作为hints传递给MultiMath"""
    
    # 类级别属性
    name = "geometry_solver"
    description = "几何问题综合求解器（DiagramFormalizer + MultiMath）"
    capabilities = ["几何问题求解", "CDL提取", "数学推理"]
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化工具"""
        super().__init__(config)
        
        self.config = config or {}
        self.df_tool = self.config.get('df_tool')
        if self.df_tool is None:
            raise ValueError("DiagramFormalizer tool is required for GeometrySolverTool")
        
        self.mm_config = self.config.get('mm_config', {
            "api_url": "http://localhost:8001",
            "timeout": 60
        })
        
        self.debug = self.config.get('debug', False)
        self.current_image = None
        self.current_image_path = None
    
    def reset(self, image=None):
        """重置工具状态"""
        if isinstance(image, str):
            self.current_image_path = image
            self.current_image = Image.open(image)
        elif isinstance(image, Image.Image):
            self.current_image = image
            self.current_image_path = getattr(image, 'filename', None)
        else:
            self.current_image = None
            self.current_image_path = None
        
        if self.df_tool and hasattr(self.df_tool, 'reset') and self.current_image:
            self.df_tool.reset(self.current_image)
    
    def execute(self, action_string) -> Dict[str, Any]:
        """执行工具"""
        if isinstance(action_string, str):
            try:
                params = json.loads(action_string)
            except json.JSONDecodeError:
                params = {"question": action_string}
        else:
            params = action_string
        
        image_path = params.get('image_path', self.current_image_path)
        question = params.get('question', params.get('problem', ''))
        
        if not question:
            return {
                "error": "Question is required",
                "success": False
            }
        
        return self.invoke(image_path, question)
    
    def invoke(self, image_path: str, question: str) -> Dict[str, Any]:
        """执行组合工具：DF → MM → Answer"""
        execution_trace = []
        df_time = 0
        mm_time = 0
        
        try:
            # Step 1: 调用DiagramFormalizer获取CDL
            if self.debug:
                print(f"\n{'='*60}")
                print(f"[GeometrySolver] Step 1: Calling DiagramFormalizer")
                print(f"{'='*60}")
                print(f"  Image: {image_path}")
                print(f"  Question: {question[:100]}...")
            
            df_start = time.time()
            
            # 确保DF有图像
            if image_path and (not self.current_image or self.current_image_path != image_path):
                image = Image.open(image_path)
                self.current_image = image
                self.current_image_path = image_path
                
                if hasattr(self.df_tool, 'reset'):
                    self.df_tool.reset(image)
                    if self.debug:
                        print(f"  ✓ DF tool reset with image")
            
            # 构建DF输入
            df_prompt = """Look at the geometric figure in the image. 
Please describe the construction and measurements by predicting the construction_cdl and image_cdl."""
            
            df_input = {
                "task": "extract_cdl",
                "problem": df_prompt
            }
            
            # 执行DF
            df_result = self.df_tool.execute(df_input)
            df_time = time.time() - df_start
            
            if self.debug:
                print(f"\n[DEBUG] DiagramFormalizer Result:")
                print(f"  - Execution time: {df_time:.2f}s")
                print(f"  - Success: {df_result.get('success', False)}")
                print(f"  - Result keys: {list(df_result.keys())}")
            
            if not df_result or not df_result.get("success", False):
                error_msg = df_result.get('error', 'Unknown error') if df_result else 'No result'
                raise RuntimeError(f"DiagramFormalizer failed: {error_msg}")
            
            # 解析CDL输出
            raw_output = (df_result.get("formalized_output", "") or 
                         df_result.get("raw_response", "") or 
                         df_result.get("output", ""))
            
            cdl_output = self._parse_cdl(raw_output)
            
            execution_trace.append({
                "tool": "diagram_formalizer",
                "time": df_time,
                "output": cdl_output
            })
            
            if self.debug:
                print(f"\n[DEBUG] Extracted CDL:")
                print(f"  ✓ CDL extracted successfully")
                print(f"  - Construction CDL: {cdl_output.get('construction_cdl', 'N/A')[:150]}...")
                print(f"  - Image CDL: {cdl_output.get('image_cdl', 'N/A')[:150]}...")
                print(f"  - Raw output length: {len(raw_output)} chars")
            
            # Step 2: 调用MultiMath Server - 简化版本，CDL作为hints
            if self.debug:
                print(f"\n{'='*60}")
                print(f"[GeometrySolver] Step 2: Calling MultiMath Server")
                print(f"{'='*60}")
                print(f"  Strategy: Original question + image + CDL hints")
            
            mm_start = time.time()
            
            # 构建MultiMath输入 - CDL作为hints
            mm_input = {
                "question": question,        # 原始问题
                "image": image_path,         # 图像路径
                "hints": {                   # CDL作为辅助信息
                    "cdl": cdl_output,
                    "source": "DiagramFormalizer"
                }
            }
            
            if self.debug:
                print(f"\n[DEBUG] MultiMath Input Structure:")
                print(f"  - Question: {question[:80]}...")
                print(f"  - Image: {os.path.basename(image_path)}")
                print(f"  - Hints structure:")
                print(f"    - CDL construction: {cdl_output.get('construction_cdl', 'N/A')[:100]}...")
                print(f"    - CDL measurements: {cdl_output.get('image_cdl', 'N/A')[:100]}...")
                print(f"    - Source: DiagramFormalizer")
            
            mm_output = self._call_multimath(mm_input)
            mm_time = time.time() - mm_start
            
            if not mm_output.get("success", False):
                raise RuntimeError(f"MultiMath failed: {mm_output.get('error', 'Unknown error')}")
            
            execution_trace.append({
                "tool": "multimath_server",
                "time": mm_time,
                "output": mm_output.get("answer", "")
            })
            
            # Step 3: 提取最终答案
            final_answer = self._extract_final_answer(mm_output)
            
            if self.debug:
                print(f"\n{'='*60}")
                print(f"[GeometrySolver] Final Results")
                print(f"{'='*60}")
                print(f"  Final answer: {final_answer}")
                print(f"  Total time: {df_time + mm_time:.2f}s")
                print(f"    - DF time: {df_time:.2f}s")
                print(f"    - MM time: {mm_time:.2f}s")
            
            return {
                "success": True,
                "final_answer": final_answer,
                "cdl": cdl_output,
                "multimath_input": mm_input,
                "multimath_output": mm_output,
                "execution_trace": execution_trace,
                "df_time": df_time,
                "mm_time": mm_time,
                "method": "DF+MM with CDL hints"
            }
            
        except Exception as e:
            if self.debug:
                print(f"\n[ERROR] GeometrySolver failed:")
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
            
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "final_answer": None,
                "cdl": {},
                "execution_trace": execution_trace,
                "df_time": df_time,
                "mm_time": mm_time
            }
    
    def _parse_cdl(self, output: str) -> Dict[str, str]:
        """解析CDL输出"""
        if not output:
            return {"construction_cdl": "", "image_cdl": ""}
        
        cdl_result = {"construction_cdl": "", "image_cdl": ""}
        
        patterns = {
            'construction_cdl': [
                r'construction_cdl[:\s]+(.*?)(?=image_cdl|$)',
                r'Construction CDL[:\s]+(.*?)(?=Image CDL|$)',
            ],
            'image_cdl': [
                r'image_cdl[:\s]+(.*?)(?=$)',
                r'Image CDL[:\s]+(.*?)(?=$)',
            ]
        }
        
        for key, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
                if match:
                    cdl_result[key] = match.group(1).strip()
                    break
        
        # 如果没找到标准格式，把整个输出作为image_cdl
        if not cdl_result["construction_cdl"] and not cdl_result["image_cdl"]:
            cdl_result["image_cdl"] = output.strip()
        
        return cdl_result
    
    def _call_multimath(self, mm_input: Dict[str, Any]) -> Dict[str, Any]:
        """调用远程MultiMath服务 - 增强调试版本"""
        try:
            if self.debug:
                print(f"\n[DEBUG] MultiMath API Call:")
                print(f"  - API URL: {self.mm_config['api_url']}/solve")
                print(f"  - Timeout: {self.mm_config.get('timeout', 60)}s")
                print(f"\n[DEBUG] Full Request Payload:")
                print("-" * 40)
                # 打印完整请求，但限制CDL长度
                debug_input = mm_input.copy()
                if 'hints' in debug_input and 'cdl' in debug_input['hints']:
                    cdl = debug_input['hints']['cdl']
                    if isinstance(cdl, dict):
                        for key in cdl:
                            if len(str(cdl[key])) > 200:
                                cdl[key] = str(cdl[key])[:200] + "..."
                print(json.dumps(debug_input, indent=2, ensure_ascii=False))
                print("-" * 40)
            
            # 发送请求
            response = requests.post(
                f"{self.mm_config['api_url']}/solve",
                json=mm_input,
                timeout=self.mm_config.get('timeout', 60)
            )
            
            if self.debug:
                print(f"\n[DEBUG] MultiMath Response:")
                print(f"  - HTTP Status: {response.status_code}")
                print(f"  - Response Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                result = response.json()
                result["success"] = True
                
                if self.debug:
                    print(f"\n[DEBUG] MultiMath Response Details:")
                    print(f"  - Success: {result.get('success', False)}")
                    print(f"  - Answer: '{result.get('answer', 'N/A')}'")
                    print(f"  - Method: {result.get('method', 'N/A')}")
                    
                    # 打印详细的求解步骤
                    if 'steps' in result and result['steps']:
                        print(f"  - Steps ({len(result['steps'])} steps):")
                        for i, step in enumerate(result['steps'][:5], 1):  # 只显示前5步
                            print(f"    {i}. {step[:100]}..." if len(str(step)) > 100 else f"    {i}. {step}")
                        if len(result['steps']) > 5:
                            print(f"    ... and {len(result['steps']) - 5} more steps")
                    else:
                        print(f"  - Steps: None or empty")
                    
                    if 'explanation' in result and result['explanation']:
                        print(f"  - Explanation: {result['explanation'][:300]}...")
                    
                    if 'raw_output' in result and result['raw_output']:
                        print(f"  - Raw output preview: {result['raw_output'][:300]}...")
                    
                    # 打印所有返回的字段
                    print(f"\n[DEBUG] All response fields:")
                    for key in result.keys():
                        if key not in ['answer', 'method', 'steps', 'explanation', 'raw_output', 'success']:
                            value = result[key]
                            if value is not None:
                                preview = str(value)[:100] if len(str(value)) > 100 else str(value)
                                print(f"    - {key}: {preview}")
                
                return result
            else:
                error_msg = f"HTTP {response.status_code}: {response.text[:500]}"
                if self.debug:
                    print(f"\n[ERROR] MultiMath API failed:")
                    print(f"  - Status: {response.status_code}")
                    print(f"  - Response: {response.text[:1000]}")
                
                return {
                    "success": False,
                    "error": error_msg,
                    "answer": None
                }
                
        except requests.exceptions.Timeout:
            error_msg = f"Request timeout after {self.mm_config.get('timeout', 60)}s"
            if self.debug:
                print(f"\n[ERROR] MultiMath API timeout: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "answer": None
            }
            
        except requests.exceptions.RequestException as e:
            if self.debug:
                print(f"\n[ERROR] MultiMath API request failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": None
            }
            
        except Exception as e:
            if self.debug:
                print(f"\n[ERROR] Unexpected error in MultiMath call: {e}")
                import traceback
                traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "answer": None
            }
    
    def _extract_final_answer(self, mm_output: Dict[str, Any]) -> Optional[str]:
        """从MultiMath输出提取最终答案"""
        if self.debug:
            print(f"\n[DEBUG] Extracting final answer:")
            print(f"  - Direct answer field: '{mm_output.get('answer', 'N/A')}'")
            print(f"  - Has solution field: {'solution' in mm_output}")
        
        # 优先使用answer字段
        if "answer" in mm_output and mm_output["answer"]:
            answer = str(mm_output["answer"])
            if self.debug:
                print(f"  - Using answer field: '{answer}'")
            return answer
        
        # 备用：从solution字段提取
        if "solution" in mm_output:
            text = str(mm_output["solution"])
            if self.debug:
                print(f"  - Trying to extract from solution: '{text[:100]}...'")
            
            match = re.search(r'(?:answer|result|y)\s*[:=]\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
            if match:
                answer = match.group(1)
                if self.debug:
                    print(f"  - Extracted from solution: '{answer}'")
                return answer
        
        if self.debug:
            print(f"  - Failed to extract answer")
        return None


class GeometrySolverTester:
    """Test GeometrySolver (DF+MM) on Geometry3K dataset"""
    
    def __init__(self, data_path: str, 
                 df_model_path: str = None,
                 mm_api_url: str = "http://localhost:8001",
                 mm_timeout: int = 60,
                 debug: bool = False):
        """Initialize tester"""
        self.data_path = data_path
        self.debug = debug
        
        # Initialize DiagramFormalizer tool
        logger.info("Initializing DiagramFormalizer tool...")
        df_config = {}
        if df_model_path:
            df_config["model_path"] = df_model_path
        if debug:
            df_config["debug"] = True
            
        try:
            self.df_tool = DiagramFormalizerTool(df_config)
            logger.info("DiagramFormalizer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize DiagramFormalizer: {e}")
            raise
        
        # Initialize GeometrySolver tool
        logger.info("Initializing GeometrySolver tool...")
        gs_config = {
            "df_tool": self.df_tool,
            "mm_config": {
                "api_url": mm_api_url,
                "timeout": mm_timeout
            },
            "debug": debug
        }
        
        try:
            self.geometry_solver = GeometrySolverTool(gs_config)
            logger.info(f"GeometrySolver initialized (MM server: {mm_api_url})")
            logger.info("CDL will be passed as hints to MultiMath")
        except Exception as e:
            logger.error(f"Failed to initialize GeometrySolver: {e}")
            raise
        
        # Load dataset
        logger.info(f"Loading dataset from {data_path}")
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        logger.info(f"Loaded {len(self.data)} problems")
        
        # Get dataset directory for image paths
        self.dataset_dir = Path(data_path).parent
        
        # Results storage
        self.results = []
        self.stats = {
            "total": 0,
            "has_image": 0,
            "solved": 0,
            "correct": 0,
            "failed": 0,
            "df_errors": 0,
            "mm_errors": 0,
            "time_total": 0,
            "df_time_total": 0,
            "mm_time_total": 0
        }
    
    def test_single_problem(self, problem_idx: int):
        """Test a single problem for debugging"""
        if problem_idx >= len(self.data):
            logger.error(f"Problem index {problem_idx} out of range")
            return
        
        problem = self.data[problem_idx]
        
        print("\n" + "="*70)
        print("TESTING SINGLE PROBLEM")
        print("="*70)
        print(f"Problem Index: {problem_idx}")
        print(f"Problem ID: {problem['id']}")
        print(f"Question: {problem['question']}")
        print(f"Ground truth: {problem['answer']}")
        print("="*70)
        
        old_debug = self.debug
        self.debug = True
        self.geometry_solver.debug = True  # 确保GeometrySolver也在debug模式
        
        result = self._test_problem(problem, problem_idx)
        
        self.debug = old_debug
        
        # Print detailed result
        print("\n" + "="*70)
        print("TEST RESULTS")
        print("="*70)
        print(f"Problem ID: {result['id']}")
        print(f"Question: {problem['question']}")
        print(f"Ground Truth: {result['ground_truth']}")
        print(f"Final Answer: {result['final_answer']}")
        print(f"Correct: {result['correct']}")
        print(f"Total Time: {result['total_time']:.3f}s")
        print(f"  - DF Time: {result['df_time']:.3f}s")
        print(f"  - MM Time: {result['mm_time']:.3f}s")
        
        if result['cdl']:
            print("\nExtracted CDL (passed as hints):")
            print(f"  Construction: {result['cdl'].get('construction_cdl', 'N/A')[:150]}...")
            print(f"  Measurements: {result['cdl'].get('image_cdl', 'N/A')[:150]}...")
        
        if result.get('mm_output'):
            print("\nMultiMath Output Summary:")
            print(f"  - Answer: {result['mm_output'].get('answer', 'N/A')}")
            print(f"  - Method: {result['mm_output'].get('method', 'N/A')}")
            print(f"  - Has steps: {'steps' in result['mm_output'] and result['mm_output']['steps']}")
        
        if result.get('error'):
            print(f"\nError: {result['error']}")
            print(f"Error Stage: {result.get('error_stage', 'unknown')}")
        
        print("="*70)
    
    def _test_problem(self, problem: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """Test a single problem"""
        result = {
            "index": idx,
            "id": problem["id"],
            "question": problem["question"],
            "ground_truth": problem["answer"],
            "has_image": False,
            "solved": False,
            "correct": False,
            "final_answer": None,
            "cdl": None,
            "mm_input": None,
            "mm_output": None,
            "total_time": 0,
            "df_time": 0,
            "mm_time": 0,
            "error": None,
            "error_stage": None
        }
        
        # Get image path
        image_path = problem.get("image_path")
        if not image_path:
            result["error"] = "No image path"
            return result
        
        # Convert to absolute path if needed
        if not Path(image_path).is_absolute():
            image_path = self.dataset_dir / image_path
        else:
            image_path = Path(image_path)
        
        # Check if image exists
        if not image_path.exists():
            result["error"] = f"Image not found: {image_path}"
            return result
        
        result["has_image"] = True
        result["image_path"] = str(image_path)
        
        # Measure total time
        start_time = time.time()
        
        try:
            # Prepare parameters for GeometrySolver
            params = {
                "image_path": str(image_path),
                "question": problem["question"]
            }
            
            if self.debug:
                print(f"\n[DEBUG] Starting test for problem {idx}")
            
            # Execute GeometrySolver
            response = self.geometry_solver.execute(params)
            
            total_time = time.time() - start_time
            result["total_time"] = total_time
            
            # Extract timing
            result["df_time"] = response.get("df_time", 0)
            result["mm_time"] = response.get("mm_time", 0)
            
            if response.get("success"):
                result["solved"] = True
                result["final_answer"] = response.get("final_answer", "")
                result["cdl"] = response.get("cdl", {})
                result["mm_input"] = response.get("multimath_input", {})
                result["mm_output"] = response.get("multimath_output", {})
                
                # Check correctness
                if result["final_answer"]:
                    result["correct"] = self._check_answer(
                        result["final_answer"], 
                        problem["answer"]
                    )
            else:
                result["error"] = response.get("error", "Unknown error")
                
                # Determine error stage
                if response.get("execution_trace"):
                    last_tool = response["execution_trace"][-1]["tool"] if response["execution_trace"] else None
                    if last_tool == "diagram_formalizer":
                        result["error_stage"] = "df"
                    elif last_tool == "multimath_server":
                        result["error_stage"] = "mm"
                    
        except Exception as e:
            result["error"] = f"Exception: {str(e)}"
            result["total_time"] = time.time() - start_time
            logger.error(f"Exception on problem {idx}: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
        
        return result
    
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
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
        
        # Try exact match first
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
    
    # ... 其他方法保持不变（run_test, _update_stats, _calculate_metrics等）


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test GeometrySolver on Geometry3K")
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/geometry3k/converted_train/geometry3k_train_vlmgym.json",
        help="Path to geometry3k JSON file"
    )
    parser.add_argument(
        "--df_model_path",
        type=str,
        default=None,
        help="Path to DiagramFormalizer model (optional)"
    )
    parser.add_argument(
        "--mm_api_url",
        type=str,
        default="http://localhost:8001",
        help="MultiMath server API URL"
    )
    parser.add_argument(
        "--mm_timeout",
        type=int,
        default=60,
        help="MultiMath server timeout in seconds"
    )
    parser.add_argument(
        "--single_problem",
        type=int,
        default=None,
        help="Test a single problem by index"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Create tester
    tester = GeometrySolverTester(
        data_path=args.data_path,
        df_model_path=args.df_model_path,
        mm_api_url=args.mm_api_url,
        mm_timeout=args.mm_timeout,
        debug=args.debug
    )
    
    # Run test
    if args.single_problem is not None:
        tester.test_single_problem(args.single_problem)


if __name__ == "__main__":
    main()