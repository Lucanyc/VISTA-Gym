#!/usr/bin/env python3
"""
Test MultiMath Server directly on Geometry3K dataset (without DiagramFormalizer)
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


class DirectMultiMathTester:
    """Test MultiMath Server directly without DiagramFormalizer"""
    
    def __init__(self, data_path: str, 
                 mm_api_url: str = "http://localhost:8001",
                 mm_timeout: int = 60,
                 debug: bool = False):
        """
        Initialize tester
        
        Args:
            data_path: Path to geometry3k JSON file
            mm_api_url: MultiMath server URL
            mm_timeout: MultiMath timeout in seconds
            debug: Enable debug mode
        """
        self.data_path = data_path
        self.debug = debug
        
        # MultiMath configuration
        self.mm_config = {
            "api_url": mm_api_url,
            "timeout": mm_timeout
        }
        
        # Check MultiMath Server status
        self._check_multimath_server()
        
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
                logger.info(f"   - Device: {health.get('device', 'unknown')}")
            else:
                logger.warning(f"⚠️ MultiMath Server response abnormal: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Cannot connect to MultiMath Server at {self.mm_config['api_url']}")
            logger.error(f"   Error: {e}")
            logger.error("   Please ensure the Docker container is running:")
            logger.error("   docker start multimath-api")
            raise
    
    def call_multimath(self, question: str, image_path: str) -> Dict[str, Any]:
        """Call MultiMath Server directly"""
        try:
            # Build request
            mm_input = {
                "question": question,
                "image": image_path,
                "problem_type": "geometry",
                "output_format": "with_steps"  # Request detailed steps
            }
            
            if self.debug:
                print(f"\n{'='*60}")
                print("[DEBUG] Calling MultiMath Server")
                print(f"{'='*60}")
                print(f"  - Question: {question[:100]}...")
                print(f"  - Image: {os.path.basename(image_path)}")
                print(f"  - API URL: {self.mm_config['api_url']}/solve")
                print(f"\n[DEBUG] Request payload:")
                print(json.dumps(mm_input, indent=2, ensure_ascii=False)[:500])
            
            # Send request
            start_time = time.time()
            response = requests.post(
                f"{self.mm_config['api_url']}/solve",
                json=mm_input,
                timeout=self.mm_config.get('timeout', 60)
            )
            elapsed_time = time.time() - start_time
            
            if self.debug:
                print(f"\n[DEBUG] Response received in {elapsed_time:.2f}s")
                print(f"  - HTTP Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                result["success"] = True
                result["time"] = elapsed_time
                
                if self.debug:
                    print(f"\n[DEBUG] MultiMath Response:")
                    print(f"  - Success: {result.get('success', False)}")
                    print(f"  - Answer: '{result.get('answer', 'N/A')}'")
                    print(f"  - Method: {result.get('method', 'N/A')}")
                    
                    if 'steps' in result and result['steps']:
                        print(f"  - Steps ({len(result['steps'])} steps):")
                        for i, step in enumerate(result['steps'][:5], 1):
                            print(f"    {i}. {step[:100]}..." if len(str(step)) > 100 else f"    {i}. {step}")
                        if len(result['steps']) > 5:
                            print(f"    ... and {len(result['steps']) - 5} more steps")
                    
                    if 'explanation' in result and result['explanation']:
                        print(f"  - Explanation: {result['explanation'][:200]}...")
                    
                    if 'raw_output' in result and result['raw_output']:
                        print(f"  - Raw output: {result['raw_output'][:200]}...")
                
                return result
            else:
                error_msg = f"HTTP {response.status_code}: {response.text[:500]}"
                if self.debug:
                    print(f"\n[ERROR] MultiMath failed: {error_msg}")
                
                return {
                    "success": False,
                    "error": error_msg,
                    "answer": None,
                    "time": elapsed_time
                }
                
        except Exception as e:
            if self.debug:
                print(f"\n[ERROR] Exception calling MultiMath: {e}")
                import traceback
                traceback.print_exc()
            
            return {
                "success": False,
                "error": str(e),
                "answer": None,
                "time": 0
            }
    
    def extract_answer(self, mm_output: Dict[str, Any]) -> Optional[str]:
        """Extract final answer from MultiMath output"""
        # Direct answer field
        if "answer" in mm_output and mm_output["answer"]:
            return str(mm_output["answer"])
        
        # From solution field
        if "solution" in mm_output:
            text = str(mm_output["solution"])
            match = re.search(r'(?:answer|result|y)\s*[:=]\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
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
    
    def test_single_problem(self, problem_idx: int):
        """Test a single problem"""
        if problem_idx >= len(self.data):
            logger.error(f"Problem index {problem_idx} out of range")
            return
        
        problem = self.data[problem_idx]
        
        print("\n" + "="*70)
        print("TESTING SINGLE PROBLEM (MultiMath Only)")
        print("="*70)
        print(f"Problem Index: {problem_idx}")
        print(f"Problem ID: {problem['id']}")
        print(f"Question: {problem['question']}")
        print(f"Ground Truth: {problem['answer']}")
        
        # Get image path
        image_path = problem.get("image_path")
        if not image_path:
            print("❌ No image path")
            return
        
        # Convert to absolute path
        if not Path(image_path).is_absolute():
            image_path = self.dataset_dir / image_path
        else:
            image_path = Path(image_path)
        
        if not image_path.exists():
            print(f"❌ Image not found: {image_path}")
            return
        
        print(f"Image: {image_path}")
        print("="*70)
        
        # Call MultiMath directly
        mm_output = self.call_multimath(problem['question'], str(image_path))
        
        # Extract answer
        final_answer = self.extract_answer(mm_output)
        
        # Check correctness
        is_correct = False
        if final_answer:
            is_correct = self.check_answer(final_answer, problem['answer'])
        
        # Print results
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"MultiMath Answer: {final_answer}")
        print(f"Ground Truth: {problem['answer']}")
        print(f"Correct: {is_correct}")
        print(f"Time: {mm_output.get('time', 0):.2f}s")
        
        if mm_output.get('steps'):
            print(f"\nSteps provided: {len(mm_output['steps'])} steps")
        
        if mm_output.get('error'):
            print(f"\nError: {mm_output['error']}")
        
        print("="*70)
        
        return {
            'problem_id': problem['id'],
            'question': problem['question'],
            'ground_truth': problem['answer'],
            'prediction': final_answer,
            'correct': is_correct,
            'time': mm_output.get('time', 0),
            'mm_output': mm_output
        }
    
    def run_test(self, max_problems: int = None, start_idx: int = 0, save_every: int = 50):
        """Run test on dataset"""
        end_idx = start_idx + max_problems if max_problems else len(self.data)
        problems = self.data[start_idx:end_idx]
        
        logger.info(f"Testing {len(problems)} problems (index {start_idx} to {end_idx-1})")
        logger.info(f"Using MultiMath Server directly (no DiagramFormalizer)")
        
        for i, problem in enumerate(tqdm(problems, desc="Testing")):
            actual_idx = start_idx + i
            result = self._test_problem(problem, actual_idx)
            self.results.append(result)
            
            self._update_stats(result)
            
            if (i + 1) % save_every == 0:
                self._save_results(intermediate=True)
                logger.info(f"Saved intermediate results after {i + 1} problems")
        
        self._calculate_metrics()
        self._save_results()
        self._print_summary()
    
    def _test_problem(self, problem: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """Test a single problem (for batch testing)"""
        result = {
            "index": idx,
            "id": problem["id"],
            "question": problem["question"],
            "ground_truth": problem["answer"],
            "has_image": False,
            "solved": False,
            "correct": False,
            "final_answer": None,
            "mm_output": None,
            "time": 0,
            "error": None
        }
        
        # Get image path
        image_path = problem.get("image_path")
        if not image_path:
            result["error"] = "No image path"
            return result
        
        # Convert to absolute path
        if not Path(image_path).is_absolute():
            image_path = self.dataset_dir / image_path
        else:
            image_path = Path(image_path)
        
        if not image_path.exists():
            result["error"] = f"Image not found: {image_path}"
            return result
        
        result["has_image"] = True
        result["image_path"] = str(image_path)
        
        try:
            # Call MultiMath
            mm_output = self.call_multimath(problem['question'], str(image_path))
            result["mm_output"] = mm_output
            result["time"] = mm_output.get('time', 0)
            
            if mm_output.get("success"):
                result["solved"] = True
                result["final_answer"] = self.extract_answer(mm_output)
                
                if result["final_answer"]:
                    result["correct"] = self.check_answer(
                        result["final_answer"], 
                        problem["answer"]
                    )
            else:
                result["error"] = mm_output.get("error", "Unknown error")
                
        except Exception as e:
            result["error"] = f"Exception: {str(e)}"
            
        return result
    
    def _update_stats(self, result: Dict[str, Any]):
        """Update statistics"""
        self.stats["total"] += 1
        
        if result["has_image"]:
            self.stats["has_image"] += 1
            
            if result["solved"]:
                self.stats["solved"] += 1
                if result["correct"]:
                    self.stats["correct"] += 1
            else:
                self.stats["failed"] += 1
        
        self.stats["time_total"] += result["time"]
    
    def _calculate_metrics(self):
        """Calculate performance metrics"""
        if self.stats["total"] == 0:
            return
        
        self.stats["accuracy"] = self.stats["correct"] / self.stats["total"] * 100
        
        if self.stats["has_image"] > 0:
            self.stats["solve_rate"] = self.stats["solved"] / self.stats["has_image"] * 100
        
        self.stats["avg_time"] = self.stats["time_total"] / self.stats["total"]
    
    def _save_results(self, intermediate: bool = False):
        """Save test results"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        output_dir = "multimath_direct_test_results"
        os.makedirs(output_dir, exist_ok=True)
        
        if intermediate:
            filename = f"intermediate_mm_direct_{len(self.results)}.json"
        else:
            filename = f"results_multimath_direct_{timestamp}.json"
        
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump({
                "test_info": {
                    "data_path": self.data_path,
                    "tool": "MultiMath Server (Direct)",
                    "timestamp": timestamp,
                    "total_problems": len(self.results)
                },
                "statistics": self.stats,
                "results": self.results
            }, f, indent=2)
        
        if not intermediate:
            logger.info(f"Results saved to {filepath}")
    
    def _print_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print("MultiMath Direct Test Summary")
        print("="*70)
        print(f"Total problems tested: {self.stats['total']}")
        print(f"Problems with images: {self.stats['has_image']}")
        print(f"Successfully solved: {self.stats['solved']}")
        print(f"Correct answers: {self.stats['correct']}")
        print(f"Failed to solve: {self.stats['failed']}")
        print(f"\nOverall accuracy: {self.stats.get('accuracy', 0):.2f}%")
        print(f"Solve rate: {self.stats.get('solve_rate', 0):.2f}%")
        print(f"Average time: {self.stats.get('avg_time', 0):.3f}s")
        print("="*70)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test MultiMath Server directly on Geometry3K")
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/geometry3k/converted_train/geometry3k_train_vlmgym.json",
        help="Path to geometry3k JSON file"
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
        "--max_problems", 
        type=int, 
        default=None,
        help="Maximum number of problems to test"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Starting index"
    )
    parser.add_argument(
        "--single_problem",
        type=int,
        default=None,
        help="Test a single problem by index"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=50,
        help="Save intermediate results every N problems"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Create tester
    tester = DirectMultiMathTester(
        data_path=args.data_path,
        mm_api_url=args.mm_api_url,
        mm_timeout=args.mm_timeout,
        debug=args.debug
    )
    
    # Run test
    if args.single_problem is not None:
        tester.test_single_problem(args.single_problem)
    else:
        tester.run_test(
            max_problems=args.max_problems,
            start_idx=args.start_idx,
            save_every=args.save_every
        )


if __name__ == "__main__":
    main()