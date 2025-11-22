"""ChartQA + ScienceQA unified evaluation script"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import json
import numpy as np
from tqdm import tqdm
from datetime import datetime

from vlm_gym.agents import VLMAgent
from vlm_gym.environments import VisionQAEnv
from vlm_gym.utils import (
    setup_logger,
    MetricsTracker,
    create_experiment_id,
    setup_experiment_directory,
    save_conversation_history
)
from data_adapters.unified_adapter import UnifiedVQAAdapter
from tasks.llava_task import LLaVATask

# Helper for JSON serialisation of NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def evaluate_unified_dataset(args):
    """Run evaluation on the unified VQA dataset."""

    # Configuration – only parameters supported by AgentConfig are included
    config = {
        "experiment": {
            "name": f"unified_vqa_eval_{args.dataset or 'all'}",
            "seed": 42
        },
        "agent": {
            # Required parameters
            "model_type": "HuggingFace",
            "model_name": args.model,

            # Generation parameters
            "max_new_tokens": 512,
            "temperature": 0.3,
            "top_p": 0.95,
            "do_sample": True,

            # Device configuration
            "device_map": "auto",
            "torch_dtype": "bfloat16",
            "trust_remote_code": True,

            # Retry configuration
            "n_retry": 3,
            "retry_delay": 1.0,

            # Advanced options
            "return_logprobs": False,
            "return_attention": False
        },
        "environment": {
            "max_steps": 3,
            "time_limit": 60
        }
    }

    # Set up experiment directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"{args.dataset or 'all'}_{timestamp}"
    exp_dirs = setup_experiment_directory("./experiments", exp_id)

    # Initialise logger
    logger = setup_logger(f"eval_{exp_id}", log_dir=exp_dirs["logs"])
    logger.info(f"Starting evaluation: {args.dataset or 'all datasets'}")

    # Create data adapter
    adapter = UnifiedVQAAdapter(
        data_root=args.data_root,
        annotation_file=args.annotation
    )

    # Show dataset statistics
    stats = adapter.get_dataset_stats()
    logger.info(f"Dataset statistics:\n{json.dumps(stats, indent=2)}")

    # Retrieve task IDs (optionally filtered by dataset)
    task_ids = adapter.get_task_ids(dataset_filter=args.dataset)

    # Limit the number of samples if requested (useful for smoke tests)
    if args.limit:
        task_ids = task_ids[:args.limit]

    logger.info(f"Will evaluate {len(task_ids)} tasks")

    # Create environment
    env = VisionQAEnv(
        dataset_path=args.data_root,
        max_steps=config["environment"]["max_steps"]
    )

    # Register the task class
    env.task_entrypoint = lambda task_id, **kwargs: LLaVATask(
        task_id=task_id,
        adapter=adapter,
        **kwargs
    )

    # Instantiate agent
    logger.info(f"Using model: {config['agent']['model_name']}")
    agent = VLMAgent(config={"agent": config["agent"]})

    # Metrics trackers
    overall_metrics = MetricsTracker()
    chartqa_metrics = MetricsTracker()
    scienceqa_metrics = MetricsTracker()

    # Store all results for later analysis
    results = []

    for task_id in tqdm(task_ids, desc="Evaluating"):
        try:
            # Fetch task metadata
            task_data = adapter.get_task_data(task_id)
            dataset_type = task_data["dataset_type"]

            logger.debug(f"Evaluating {task_id} ({dataset_type})")

            # Reset environment for this task
            obs, info = env.reset(task_id=task_id)

            # Ensure observation contains image_path
            if "task_info" in obs and "image_path" in info["task_info"]:
                # Build observation dictionary expected by VLMAgent
                agent_obs = {
                    "image_path": info["task_info"]["image_path"],
                    "question": obs.get("text", info["task_goal"]),
                    "choices": None  # Add choices here if the task is MCQ
                }
            else:
                # Fallback: obtain information from task_data
                agent_obs = {
                    "image_path": task_data["image_path"],
                    "question": task_data["question"],
                    "choices": None
                }

            done = False
            step_count = 0
            total_reward = 0
            conversation_history = []
            user_answer = ""  # final answer produced by agent

            while not done and step_count < config["environment"]["max_steps"]:
                # Agent produces an action based on the current observation
                action, extra_info = agent.act(agent_obs)

                # If the action is plain text, wrap it in the env-required format
                # The environment expects something like "answer_question(...)"
                if not action.startswith("answer_question") and not action.startswith("Error:"):
                    user_answer = action  # keep a copy of the raw answer
                    formatted_action = f'answer_question(question="{agent_obs["question"]}", answer="{action}")'
                else:
                    formatted_action = action
                    # Extract answer string from the formatted action for bookkeeping
                    import re
                    match = re.search(r'answer="([^"]*)"', action)
                    if match:
                        user_answer = match.group(1)

                # Log the step for later analysis
                conversation_history.append({
                    "step": step_count,
                    "observation": agent_obs,
                    "action": action,
                    "formatted_action": formatted_action,
                    "extra_info": extra_info
                })

                # Execute the action in the environment
                try:
                    obs, reward, done, truncated, info = env.step(formatted_action)
                except Exception as e:
                    logger.debug(f"Action execution failed: {e}")
                    # If execution fails, end the episode with zero reward
                    done = True
                    reward = 0.0
                    info = {
                        "user_answer": user_answer,
                        "correct": False,
                        "error": str(e)
                    }

                total_reward += reward
                step_count += 1

                if done or truncated:
                    break

                # Prepare observation for the next step
                if "task_info" in obs and "image_path" in obs["task_info"]:
                    agent_obs = {
                        "image_path": obs["task_info"]["image_path"],
                        "question": obs.get("text", ""),
                        "choices": None
                    }

            # Aggregate results for this task
            result = {
                "task_id": task_id,
                "dataset_type": dataset_type,
                "question": task_data["question"],
                "ground_truth": task_data["answer"],
                "prediction": user_answer or info.get("user_answer", ""),
                "correct": info.get("correct", False),
                "reward": float(total_reward),  # cast to float for JSON serialisation
                "steps": int(step_count),
                "timestamp": datetime.now().isoformat()
            }

            # Basic post-hoc evaluation if the environment did not mark correctness
            if not result["correct"] and result["prediction"] and result["ground_truth"]:
                pred_lower = result["prediction"].lower()
                truth_lower = result["ground_truth"].lower()

                # Numeric answers – check if any ground-truth number appears in prediction
                import re
                pred_numbers = re.findall(r'\d+\.?\d*', result["prediction"])
                truth_numbers = re.findall(r'\d+\.?\d*', result["ground_truth"])

                if truth_numbers and pred_numbers:
                    if any(num in pred_numbers for num in truth_numbers):
                        result["correct"] = True
                        result["reward"] = 1.0
                elif truth_lower in pred_lower or pred_lower in truth_lower:
                    result["correct"] = True
                    result["reward"] = 1.0

            results.append(result)

            # Update metrics
            overall_metrics.add("accuracy", float(result["correct"]))
            overall_metrics.add("reward", float(result["reward"]))
            overall_metrics.add("steps", float(step_count))

            # Dataset-specific metrics
            if dataset_type == "chartqa":
                chartqa_metrics.add("accuracy", float(result["correct"]))
                chartqa_metrics.add("reward", float(result["reward"]))
            elif dataset_type == "scienceqa":
                scienceqa_metrics.add("accuracy", float(result["correct"]))
                scienceqa_metrics.add("reward", float(result["reward"]))

            # Optionally save conversation history for qualitative inspection
            if args.save_conversations:
                save_conversation_history(
                    conversation_history,
                    exp_dirs["results"] / f"{task_id}_conversation.json"
                )

        except Exception as e:
            logger.error(f"Error on task {task_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append({
                "task_id": task_id,
                "error": str(e),
                "success": False
            })

    # Persist detailed results
    results_file = exp_dirs["results"] / "detailed_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Detailed results saved to: {results_file}")

    # Print summary report to console
    print("\n" + "="*70)
    print("EVALUATION REPORT")
    print("="*70)
    print(f"Experiment ID: {exp_id}")
    print(f"Total tasks evaluated: {len(results)}")
    print(f"\nOVERALL METRICS:")
    print(f"  Accuracy: {overall_metrics.get_mean('accuracy'):.2%}")
    print(f"  Average reward: {overall_metrics.get_mean('reward'):.3f}")
    print(f"  Average steps: {overall_metrics.get_mean('steps'):.1f}")

    if chartqa_metrics.metrics:
        print(f"\nCHARTQA METRICS:")
        print(f"  Samples: {len(chartqa_metrics.metrics['accuracy'])}")
        print(f"  Accuracy: {chartqa_metrics.get_mean('accuracy'):.2%}")
        print(f"  Average reward: {chartqa_metrics.get_mean('reward'):.3f}")

    if scienceqa_metrics.metrics:
        print(f"\nSCIENCEQA METRICS:")
        print(f"  Samples: {len(scienceqa_metrics.metrics['accuracy'])}")
        print(f"  Accuracy: {scienceqa_metrics.get_mean('accuracy'):.2%}")
        print(f"  Average reward: {scienceqa_metrics.get_mean('reward'):.3f}")

    print("="*70)

    # Show a few sample predictions
    print("\nSample predictions:")
    for i, result in enumerate(results[:3]):  # show the first three results
        if "error" not in result:
            print(f"\nTask {i+1}:")
            print(f"  Question: {result['question']}")
            print(f"  Ground Truth: {result['ground_truth']}")
            print(f"  Prediction: {result['prediction'][:100]}...")  # truncate long answers
            print(f"  Correct: {result['correct']}")

    # Summarise and save overview JSON
    summary = {
        "experiment_id": exp_id,
        "config": config,
        "dataset_stats": stats,
        "overall_metrics": overall_metrics.get_summary(),
        "chartqa_metrics": chartqa_metrics.get_summary() if chartqa_metrics.metrics else {},
        "scienceqa_metrics": scienceqa_metrics.get_summary() if scienceqa_metrics.metrics else {},
        "num_evaluated": len(results),
        "num_errors": sum(1 for r in results if "error" in r)
    }

    summary_file = exp_dirs["results"] / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    print(f"\nSummary saved to: {summary_file}")

    # Save error analysis for incorrect predictions
    errors = [r for r in results if not r.get("correct", False) and "error" not in r]
    if errors:
        error_analysis_file = exp_dirs["results"] / "error_analysis.json"
        with open(error_analysis_file, 'w') as f:
            json.dump(errors[:10], f, indent=2, cls=NumpyEncoder)
        print(f"Error analysis saved to: {error_analysis_file}")

    env.close()

    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate unified VQA dataset")
    parser.add_argument("--annotation", type=str,
                        default="/GYM-Work/try_vlm_gym/data/vision_r1_sample_dataset.json",
                        help="Path to annotation file")
    parser.add_argument("--data-root", type=str,
                        default="/GYM-Work/try_vlm_gym/data",
                        help="Root directory for images")
    parser.add_argument("--dataset", type=str, choices=["chartqa", "scienceqa"],
                        default=None, help="Evaluate specific dataset only")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="Model to use for evaluation")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples to evaluate")
    parser.add_argument("--save-conversations", action="store_true",
                        help="Save conversation history for each task")

    args = parser.parse_args()

    evaluate_unified_dataset(args)

if __name__ == "__main__":
    main()
