#!/usr/bin/env python3
"""
Test MultiMath model on Geometry3K dataset
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
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add VLM-Gym to path
sys.path.append('/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista')

# Import MultiMath tool
from vlm_gym.environments.tools.multimath import MultiMathTool


class MultiMathGeometry3KTester:
    """Test MultiMath on Geometry3K dataset"""
    
    def __init__(self, data_path: str, model_path: str = "./checkpoints/multimath-7b-llava-v1.5",
                 task: str = "solve", output_format: str = "answer_only",
                 temperature: float = 0.0, max_new_tokens: int = 512,
                 debug: bool = False, device: str = None):
        """
        Initialize tester
        
        Args:
            data_path: Path to geometry3k JSON file
            model_path: Path to MultiMath model checkpoint
            task: Task type (solve, analyze, explain)
            output_format: Output format (answer_only, with_steps, detailed)
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
            debug: Enable debug mode
            device: Device to use (cuda/cpu)
        """
        self.data_path = data_path
        self.task = task
        self.output_format = output_format
        self.debug = debug
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize MultiMath tool
        logger.info(f"Initializing MultiMath from {model_path}")
        self.tool = MultiMathTool(config={
            "model_path": model_path,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "debug": debug,
            "device": self.device
        })
        
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
            "errors": 0,
            "time_total": 0,
            "inference_time_total": 0
        }
    
    def run_test(self, max_problems: int = None, start_idx: int = 0, save_every: int = 50):
        """
        Run test on dataset
        
        Args:
            max_problems: Maximum number of problems to test
            start_idx: Starting index
            save_every: Save intermediate results every N problems
        """
        # Select problems to test
        end_idx = start_idx + max_problems if max_problems else len(self.data)
        problems = self.data[start_idx:end_idx]
        
        logger.info(f"Testing {len(problems)} problems (index {start_idx} to {end_idx-1})")
        logger.info(f"Task: {self.task}, Output format: {self.output_format}")
        
        # Test each problem
        for i, problem in enumerate(tqdm(problems, desc="Testing")):
            actual_idx = start_idx + i
            result = self._test_problem(problem, actual_idx)
            self.results.append(result)
            
            # Update statistics
            self._update_stats(result)
            
            # Save intermediate results
            if (i + 1) % save_every == 0:
                self._save_results(intermediate=True)
                logger.info(f"Saved intermediate results after {i + 1} problems")
        
        # Calculate final metrics
        self._calculate_metrics()
        
        # Save final results
        self._save_results()
        
        # Print summary
        self._print_summary()
    
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
            "multimath_answer": None,
            "multimath_steps": None,
            "multimath_analysis": None,
            "total_time": 0,
            "inference_time": 0,
            "error": None
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
        
        # Load and set image for the tool
        try:
            image = Image.open(image_path)
            self.tool.reset(image)
        except Exception as e:
            result["error"] = f"Failed to load image: {str(e)}"
            return result
        
        # Measure total time
        start_time = time.time()
        
        try:
            # Prepare parameters
            params = {
                "task": self.task,
                "question": problem["question"],
                "image": str(image_path),
                "problem_type": "geometry",  # Geometry3K is geometry dataset
                "output_format": self.output_format
            }
            
            # Execute MultiMath
            if self.debug:
                logger.debug(f"Testing problem {idx}: {problem['question'][:50]}...")
            
            # Time the inference
            inference_start = time.time()
            response = self.tool.execute(params)
            inference_time = time.time() - inference_start
            
            total_time = time.time() - start_time
            result["total_time"] = total_time
            result["inference_time"] = inference_time
            
            if response.get("success"):
                result["solved"] = True
                
                # Extract results based on task type
                if self.task == "solve":
                    result["multimath_answer"] = response.get("answer", "")
                    if self.output_format != "answer_only":
                        result["multimath_steps"] = response.get("steps", [])
                elif self.task == "analyze":
                    result["multimath_analysis"] = response.get("analysis", {})
                elif self.task == "explain":
                    result["multimath_explanation"] = response.get("explanation", {})
                
                # Store raw response if detailed output
                if self.output_format == "detailed":
                    result["full_response"] = response.get("full_response", "")
                
                # Check correctness for solve task
                if self.task == "solve" and result["multimath_answer"]:
                    result["correct"] = self._check_answer(
                        result["multimath_answer"], 
                        problem["answer"]
                    )
                
                if self.debug:
                    logger.debug(f"Problem {idx}: MultiMath={result['multimath_answer']}, "
                               f"GT={problem['answer']}, Correct={result['correct']}")
            else:
                result["error"] = response.get("error", "Unknown error")
                if self.debug:
                    logger.debug(f"Problem {idx} failed: {result['error']}")
                    
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
            # Extract numeric value
            import re
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
                if result.get("error"):
                    self.stats["errors"] += 1
        
        self.stats["time_total"] += result["total_time"]
        self.stats["inference_time_total"] += result["inference_time"]
    
    def _calculate_metrics(self):
        """Calculate performance metrics"""
        if self.stats["total"] == 0:
            return
        
        # Overall accuracy
        self.stats["accuracy"] = self.stats["correct"] / self.stats["total"] * 100
        
        # Solve rate (among problems with images)
        if self.stats["has_image"] > 0:
            self.stats["solve_rate"] = self.stats["solved"] / self.stats["has_image"] * 100
            self.stats["solve_accuracy"] = self.stats["correct"] / self.stats["has_image"] * 100
        
        # Average times
        self.stats["avg_total_time"] = self.stats["time_total"] / self.stats["total"]
        self.stats["avg_inference_time"] = self.stats["inference_time_total"] / self.stats["total"]
        
        # Problem type analysis
        self._analyze_by_type()
        
        # Error analysis
        self._analyze_errors()
    
    def _analyze_by_type(self):
        """Analyze results by problem type"""
        type_stats = {}
        
        for i, result in enumerate(self.results):
            if i < len(self.data):
                problem = self.data[i]
                metadata = problem.get("metadata", {})
                
                # Analyze by graph type
                graph_types = metadata.get("problem_type_graph", ["Unknown"])
                for gtype in graph_types:
                    if gtype not in type_stats:
                        type_stats[gtype] = {"total": 0, "correct": 0, "solved": 0}
                    
                    type_stats[gtype]["total"] += 1
                    if result["solved"]:
                        type_stats[gtype]["solved"] += 1
                    if result["correct"]:
                        type_stats[gtype]["correct"] += 1
        
        # Calculate accuracy for each type
        for gtype, stats in type_stats.items():
            if stats["total"] > 0:
                stats["accuracy"] = stats["correct"] / stats["total"] * 100
                stats["solve_rate"] = stats["solved"] / stats["total"] * 100
        
        self.stats["by_type"] = type_stats
    
    def _analyze_errors(self):
        """Analyze error types"""
        error_types = {}
        
        for result in self.results:
            if result.get("error"):
                error_msg = result["error"]
                
                # Categorize errors
                if "Image not found" in error_msg:
                    error_type = "missing_image"
                elif "Failed to load image" in error_msg:
                    error_type = "image_load_error"
                elif "CUDA out of memory" in error_msg:
                    error_type = "out_of_memory"
                elif "Exception:" in error_msg:
                    error_type = "exception"
                else:
                    error_type = "other"
                
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        self.stats["error_types"] = error_types
    
    def _save_results(self, intermediate: bool = False):
        """Save test results"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        output_dir = "multimath_test_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare filename
        if intermediate:
            filename = f"intermediate_{self.task}_{self.output_format}_{len(self.results)}.json"
        else:
            filename = f"results_{self.task}_{self.output_format}_{timestamp}.json"
        
        filepath = os.path.join(output_dir, filename)
        
        # Save results
        with open(filepath, 'w') as f:
            json.dump({
                "test_info": {
                    "data_path": self.data_path,
                    "task": self.task,
                    "output_format": self.output_format,
                    "timestamp": timestamp,
                    "total_problems": len(self.results),
                    "device": self.device
                },
                "statistics": self.stats,
                "results": self.results
            }, f, indent=2)
        
        if not intermediate:
            logger.info(f"Results saved to {filepath}")
    
    def _print_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print(f"MultiMath Test Summary - Task: {self.task}, Format: {self.output_format}")
        print("="*70)
        print(f"Total problems tested: {self.stats['total']}")
        print(f"Problems with images: {self.stats['has_image']}")
        print(f"Successfully solved: {self.stats['solved']}")
        print(f"Correct answers: {self.stats['correct']}")
        print(f"Failed to solve: {self.stats['failed']}")
        print(f"Errors encountered: {self.stats['errors']}")
        print(f"\nOverall accuracy: {self.stats.get('accuracy', 0):.2f}%")
        print(f"Solve rate: {self.stats.get('solve_rate', 0):.2f}%")
        print(f"Solve accuracy: {self.stats.get('solve_accuracy', 0):.2f}%")
        print(f"Average total time: {self.stats.get('avg_total_time', 0):.3f}s")
        print(f"Average inference time: {self.stats.get('avg_inference_time', 0):.3f}s")
        
        # Print results by type
        if "by_type" in self.stats:
            print("\n" + "-"*50)
            print("Results by Problem Type:")
            print("-"*50)
            for ptype, stats in sorted(self.stats["by_type"].items()):
                print(f"{ptype:15} - Total: {stats['total']:4d}, "
                      f"Solved: {stats['solved']:4d} ({stats['solve_rate']:.1f}%), "
                      f"Correct: {stats['correct']:4d} ({stats['accuracy']:.1f}%)")
        
        # Print error analysis
        if "error_types" in self.stats:
            print("\n" + "-"*50)
            print("Error Analysis:")
            print("-"*50)
            for error_type, count in sorted(self.stats["error_types"].items()):
                print(f"{error_type:20} - {count:4d}")
        
        print("="*70)
    
    def test_single_problem(self, problem_idx: int):
        """Test a single problem for debugging"""
        if problem_idx >= len(self.data):
            logger.error(f"Problem index {problem_idx} out of range")
            return
        
        problem = self.data[problem_idx]
        logger.info(f"Testing single problem {problem_idx}: {problem['id']}")
        logger.info(f"Question: {problem['question']}")
        logger.info(f"Ground truth: {problem['answer']}")
        
        # Enable debug for single problem test
        old_debug = self.debug
        self.debug = True
        
        result = self._test_problem(problem, problem_idx)
        
        self.debug = old_debug
        
        # Print detailed result
        print("\n" + "="*50)
        print("Single Problem Test Result:")
        print("="*50)
        print(f"Problem ID: {result['id']}")
        print(f"Question: {problem['question']}")
        print(f"Ground Truth: {result['ground_truth']}")
        print(f"MultiMath Answer: {result['multimath_answer']}")
        print(f"Correct: {result['correct']}")
        print(f"Inference Time: {result['inference_time']:.3f}s")
        
        if result['multimath_steps']:
            print("\nSteps:")
            for i, step in enumerate(result['multimath_steps']):
                print(f"{i+1}. {step}")
        
        if result.get('error'):
            print(f"\nError: {result['error']}")
        
        print("="*50)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test MultiMath on Geometry3K")
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/geometry3k/converted_train/geometry3k_train_vlmgym.json",
        help="Path to geometry3k JSON file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./checkpoints/multimath-7b-llava-v1.5",
        help="Path to MultiMath model checkpoint"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="solve",
        choices=["solve", "analyze", "explain"],
        help="Task type"
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="answer_only",
        choices=["answer_only", "with_steps", "detailed"],
        help="Output format"
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
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use"
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
    tester = MultiMathGeometry3KTester(
        data_path=args.data_path,
        model_path=args.model_path,
        task=args.task,
        output_format=args.output_format,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        debug=args.debug,
        device=args.device
    )
    
    # Run test
    if args.single_problem is not None:
        # Test single problem
        tester.test_single_problem(args.single_problem)
    else:
        # Run full test
        tester.run_test(
            max_problems=args.max_problems,
            start_idx=args.start_idx,
            save_every=args.save_every
        )


if __name__ == "__main__":
    main()