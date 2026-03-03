#!/usr/bin/env python3
"""
Test Inter-GPS tool on Geometry3K dataset in VLM-Gym environment
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add VLM-Gym to path
sys.path.append('/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista')

# Import Inter-GPS tool
from vlm_gym.environments.tools.intergps import InterGPSTool


class InterGPSGeometry3KTester:
    """Test Inter-GPS on Geometry3K dataset"""
    
    def __init__(self, data_path: str, strategy: str = "final", 
                 time_limit: int = 100, num_threads: int = 4, debug: bool = False):
        """
        Initialize tester
        
        Args:
            data_path: Path to geometry3k JSON file
            strategy: Inter-GPS search strategy
            time_limit: Time limit per problem
            num_threads: Number of threads for Inter-GPS
            debug: Enable debug mode
        """
        self.data_path = data_path
        self.strategy = strategy
        self.debug = debug
        
        # Initialize Inter-GPS tool
        self.tool = InterGPSTool(config={
            "strategy": strategy,
            "time_limit": time_limit,
            "num_threads": num_threads,
            "debug": debug
        })
        
        # Load dataset
        logger.info(f"Loading dataset from {data_path}")
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        logger.info(f"Loaded {len(self.data)} problems")
        
        # Results storage
        self.results = []
        self.stats = {
            "total": 0,
            "has_logic_form": 0,
            "solved": 0,
            "correct": 0,
            "failed": 0,
            "errors": 0,
            "time_total": 0
        }
    
    def run_test(self, max_problems: int = None, start_idx: int = 0):
        """
        Run test on dataset
        
        Args:
            max_problems: Maximum number of problems to test
            start_idx: Starting index
        """
        # Select problems to test
        end_idx = start_idx + max_problems if max_problems else len(self.data)
        problems = self.data[start_idx:end_idx]
        
        logger.info(f"Testing {len(problems)} problems (index {start_idx} to {end_idx-1})")
        logger.info(f"Using strategy: {self.strategy}")
        
        # Test each problem
        for i, problem in enumerate(tqdm(problems, desc="Testing")):
            actual_idx = start_idx + i
            result = self._test_problem(problem, actual_idx)
            self.results.append(result)
            
            # Update statistics
            self._update_stats(result)
            
            # Save intermediate results every 50 problems
            if (i + 1) % 50 == 0:
                self._save_results(intermediate=True)
        
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
            "has_logic_form": False,
            "solved": False,
            "correct": False,
            "intergps_answer": None,
            "solve_time": 0,
            "error": None
        }
        
        # Check if logic forms exist
        metadata = problem.get("metadata", {})
        text_logic = metadata.get("text_logic_form", [])
        diagram_logic = metadata.get("diagram_logic_form", [])
        
        if not text_logic or not diagram_logic:
            result["error"] = "Missing logic forms"
            return result
        
        result["has_logic_form"] = True
        
        # Set current problem context
        self.tool.set_current_problem(problem)
        
        # Measure solve time
        start_time = time.time()
        
        try:
            # Execute Inter-GPS
            response = self.tool.execute({
                "task": "solve",
                "problem_id": problem["id"],
                "text_logic": text_logic,
                "diagram_logic": diagram_logic,
                "strategy": self.strategy
            })
            
            solve_time = time.time() - start_time
            result["solve_time"] = solve_time
            
            if response.get("success"):
                result["solved"] = True
                result["intergps_answer"] = response.get("solution")
                result["method"] = response.get("method")
                result["find_target"] = response.get("find_target")
                
                # Check correctness
                result["correct"] = self._check_answer(
                    result["intergps_answer"], 
                    problem["answer"]
                )
                
                if self.debug:
                    logger.debug(f"Problem {idx}: GPS={result['intergps_answer']}, "
                               f"GT={problem['answer']}, Correct={result['correct']}")
            else:
                result["error"] = response.get("error", "Unknown error")
                if self.debug:
                    logger.debug(f"Problem {idx} failed: {result['error']}")
                    
        except Exception as e:
            result["error"] = f"Exception: {str(e)}"
            result["solve_time"] = time.time() - start_time
            logger.error(f"Exception on problem {idx}: {e}")
        
        return result
    
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth"""
        if predicted is None or ground_truth is None:
            return False
        
        # Clean answers
        pred_clean = str(predicted).strip()
        gt_clean = str(ground_truth).strip()
        
        # Try exact match first
        if pred_clean == gt_clean:
            return True
        
        # Try numeric comparison
        try:
            pred_num = float(pred_clean)
            gt_num = float(gt_clean)
            
            # Check with tolerance
            abs_tol = 0.01
            rel_tol = 1e-9
            
            return (abs(pred_num - gt_num) < abs_tol or 
                   abs(pred_num - gt_num) / max(abs(pred_num), abs(gt_num), 1) < rel_tol)
        except:
            return False
    
    def _update_stats(self, result: Dict[str, Any]):
        """Update statistics"""
        self.stats["total"] += 1
        
        if result["has_logic_form"]:
            self.stats["has_logic_form"] += 1
            
            if result["solved"]:
                self.stats["solved"] += 1
                if result["correct"]:
                    self.stats["correct"] += 1
            else:
                self.stats["failed"] += 1
                if result.get("error"):
                    self.stats["errors"] += 1
        
        self.stats["time_total"] += result["solve_time"]
    
    def _calculate_metrics(self):
        """Calculate performance metrics"""
        if self.stats["total"] == 0:
            return
        
        # Overall accuracy
        self.stats["accuracy"] = self.stats["correct"] / self.stats["total"] * 100
        
        # Solve rate (among problems with logic forms)
        if self.stats["has_logic_form"] > 0:
            self.stats["solve_rate"] = self.stats["solved"] / self.stats["has_logic_form"] * 100
            self.stats["solve_accuracy"] = self.stats["correct"] / self.stats["has_logic_form"] * 100
        
        # Average solve time
        self.stats["avg_solve_time"] = self.stats["time_total"] / self.stats["total"]
        
        # Problem type analysis
        self._analyze_by_type()
    
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
    
    def _save_results(self, intermediate: bool = False):
        """Save test results"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        output_dir = "intergps_test_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare filename
        if intermediate:
            filename = f"intermediate_{self.strategy}_{len(self.results)}.json"
        else:
            filename = f"results_{self.strategy}_{timestamp}.json"
        
        filepath = os.path.join(output_dir, filename)
        
        # Save results
        with open(filepath, 'w') as f:
            json.dump({
                "test_info": {
                    "data_path": self.data_path,
                    "strategy": self.strategy,
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
        print(f"Inter-GPS Test Summary - Strategy: {self.strategy}")
        print("="*70)
        print(f"Total problems tested: {self.stats['total']}")
        print(f"Problems with logic forms: {self.stats['has_logic_form']}")
        print(f"Successfully solved: {self.stats['solved']}")
        print(f"Correct answers: {self.stats['correct']}")
        print(f"Failed to solve: {self.stats['failed']}")
        print(f"Errors encountered: {self.stats['errors']}")
        print(f"\nOverall accuracy: {self.stats.get('accuracy', 0):.2f}%")
        print(f"Solve rate: {self.stats.get('solve_rate', 0):.2f}%")
        print(f"Solve accuracy: {self.stats.get('solve_accuracy', 0):.2f}%")
        print(f"Average solve time: {self.stats.get('avg_solve_time', 0):.3f}s")
        
        # Print results by type
        if "by_type" in self.stats:
            print("\n" + "-"*50)
            print("Results by Problem Type:")
            print("-"*50)
            for ptype, stats in sorted(self.stats["by_type"].items()):
                print(f"{ptype:15} - Total: {stats['total']:4d}, "
                      f"Solved: {stats['solved']:4d} ({stats['solve_rate']:.1f}%), "
                      f"Correct: {stats['correct']:4d} ({stats['accuracy']:.1f}%)")
        
        print("="*70)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test Inter-GPS on Geometry3K")
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/geometry3k/converted_train/geometry3k_train_vlmgym.json",
        help="Path to geometry3k JSON file"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="final",
        choices=["final", "predict", "random", "low-first"],
        help="Inter-GPS search strategy"
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
        "--time_limit",
        type=int,
        default=100,
        help="Time limit per problem (seconds)"
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=4,
        help="Number of threads for Inter-GPS"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Create tester
    tester = InterGPSGeometry3KTester(
        data_path=args.data_path,
        strategy=args.strategy,
        time_limit=args.time_limit,
        num_threads=args.num_threads,
        debug=args.debug
    )
    
    # Run test
    tester.run_test(
        max_problems=args.max_problems,
        start_idx=args.start_idx
    )


if __name__ == "__main__":
    main()