import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Compute scores for VLM experiments")
    parser.add_argument("--results_path", type=str, required=True,
                        help="Path to results directory")
    parser.add_argument("--metrics", nargs="+", default=["accuracy", "reward"],
                        help="Metrics to compute")
    parser.add_argument("--group_by", type=str, default=None,
                        help="Group results by this field (e.g., 'task_type')")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for detailed results")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information")
    return parser.parse_args()

def load_results(results_path: str) -> List[Dict[str, Any]]:
    """Load all result files from directory"""
    results = []
    path = Path(results_path)
    
    # Support both single file and directory
    if path.is_file():
        with open(path, 'r') as f:
            results.append(json.load(f))
    else:
        for file_path in path.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    data['file_name'] = file_path.name
                    results.append(data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
    return results

def compute_basic_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute basic success/failure metrics"""
    metrics = {
        "total_episodes": len(results),
        "successful_episodes": 0,
        "failed_episodes": 0,
        "truncated_episodes": 0,
        "avg_steps": 0,
        "avg_reward": 0,
    }
    
    total_steps = 0
    total_reward = 0
    
    for result in results:
        # Check final status
        if isinstance(result, list) and len(result) > 0:
            final_state = result[-1]
        else:
            final_state = result
            
        if final_state.get("result") == "success" or final_state.get("done", False):
            metrics["successful_episodes"] += 1
        elif final_state.get("truncated", False):
            metrics["truncated_episodes"] += 1
        else:
            metrics["failed_episodes"] += 1
            
        # Compute steps and rewards
        if "total_steps" in final_state:
            total_steps += final_state["total_steps"]
        elif isinstance(result, list):
            total_steps += len(result)
            
        if "total_reward" in final_state:
            total_reward += final_state["total_reward"]
        elif isinstance(result, list):
            total_reward += sum(step.get("reward", 0) for step in result)
            
    metrics["avg_steps"] = total_steps / max(1, metrics["total_episodes"])
    metrics["avg_reward"] = total_reward / max(1, metrics["total_episodes"])
    metrics["success_rate"] = metrics["successful_episodes"] / max(1, metrics["total_episodes"])
    
    return metrics

def compute_vision_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute vision-specific metrics"""
    vision_metrics = {
        "image_analysis_accuracy": 0,
        "region_description_accuracy": 0,
        "visual_grounding_success": 0,
        "multi_image_comparison_accuracy": 0,
    }
    
    # Count vision-related actions
    action_counts = defaultdict(int)
    action_success = defaultdict(int)
    
    for result in results:
        if isinstance(result, list):
            for step in result:
                if "action" in step:
                    action_type = step.get("action_type", "unknown")
                    action_counts[action_type] += 1
                    if step.get("success", False):
                        action_success[action_type] += 1
                        
    # Compute accuracies
    for action_type in ["analyze_image", "describe_region", "visual_grounding", "compare_images"]:
        if action_counts[action_type] > 0:
            key = f"{action_type}_accuracy"
            if key in vision_metrics:
                vision_metrics[key] = action_success[action_type] / action_counts[action_type]
                
    return vision_metrics

def group_results(results: List[Dict[str, Any]], group_by: str) -> Dict[str, List[Dict]]:
    """Group results by a specific field"""
    grouped = defaultdict(list)
    
    for result in results:
        # Extract grouping value
        if isinstance(result, list) and len(result) > 0:
            group_value = result[0].get(group_by, "unknown")
        else:
            group_value = result.get(group_by, "unknown")
            
        grouped[group_value].append(result)
        
    return dict(grouped)

def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """Pretty print metrics"""
    print(f"\n{title}")
    print("=" * 50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:.<40} {value:.4f}")
        else:
            print(f"{key:.<40} {value}")

def main():
    args = parse_args()
    
    # Load results
    results = load_results(args.results_path)
    print(f"Loaded {len(results)} result files")
    
    # Compute overall metrics
    basic_metrics = compute_basic_metrics(results)
    print_metrics(basic_metrics, "Basic Metrics")
    
    # Compute vision-specific metrics
    if "vision" in args.metrics:
        vision_metrics = compute_vision_metrics(results)
        print_metrics(vision_metrics, "Vision Metrics")
    
    # Group results if requested
    if args.group_by:
        grouped_results = group_results(results, args.group_by)
        print(f"\nGrouped by {args.group_by}:")
        
        for group_name, group_results in grouped_results.items():
            group_metrics = compute_basic_metrics(group_results)
            print_metrics(group_metrics, f"Group: {group_name}")
    
    # Save detailed results if requested
    if args.output:
        all_metrics = {
            "basic": basic_metrics,
            "vision": vision_metrics if "vision" in args.metrics else {},
            "grouped": {}
        }
        
        if args.group_by:
            for group_name, group_results in grouped_results.items():
                all_metrics["grouped"][group_name] = compute_basic_metrics(group_results)
                
        with open(args.output, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nDetailed results saved to {args.output}")

if __name__ == "__main__":
    main()
