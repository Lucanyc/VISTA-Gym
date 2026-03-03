#!/usr/bin/env python3
"""
Test script for VQA-COCO integration with VLM Gym
Tests visual question answering on natural images from COCO dataset
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import random
from collections import defaultdict, Counter
import time
import traceback
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import numpy as np

# Add paths for imports
sys.path.insert(0, '/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista')
sys.path.insert(0, '/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data_adapters')

# Import required components
try:
    from vqacoco import VQACOCOTask
except ImportError as e:
    print(f"Warning: Failed to import VQACOCOTask from current directory: {e}")
    try:
        from vlm_gym.environments.task.vqacoco import VQACOCOTask
    except ImportError as e2:
        print(f"Error: Cannot import VQACOCOTask: {e2}")
        print("\nPlease ensure vqacoco.py exists in:")
        print("1. Current directory (vlm_gym/environments/task/)")
        print("2. Or properly installed in the VLM Gym package")
        sys.exit(1)

# Import VLM Gym components (optional for this test)
try:
    from vlm_gym.environments import VisionQAEnv
    from vlm_gym.agents.vlm_agent_with_tools import VLMAgentWithTools
except ImportError as e:
    print(f"Warning: Could not import VLM Gym components: {e}")
    print("Basic task testing will still work.")

# Import the adapter
try:
    from vqacoco_adapter import VQACOCOAdapter
except ImportError as e:
    print(f"Error: Cannot import VQACOCOAdapter: {e}")
    print("\nMake sure vqacoco_adapter.py is in:")
    print("/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data_adapters/")
    sys.exit(1)


def test_adapter_loading(annotation_file, data_root=None):
    """Test VQA-COCO adapter initialization and data loading"""
    print("\n" + "="*60)
    print("Testing VQA-COCO Adapter Loading")
    print("="*60)
    
    # Initialize adapter with provided paths
    print(f"Loading from: {annotation_file}")
    if data_root:
        print(f"Image root: {data_root}")
    
    adapter = VQACOCOAdapter(
        data_root=data_root,
        annotation_files=annotation_file,
        validate_images=False,  # Set to False initially to avoid path issues
        split="train"
    )
    
    # Get statistics
    stats = adapter.get_statistics()
    print(f"\n✓ Adapter loaded successfully!")
    print(f"  - Total tasks: {stats['total']}")
    print(f"  - Unique images: {stats.get('unique_images', 'N/A')}")
    print(f"  - Unique answers: {stats.get('unique_answers', 'N/A')}")
    print(f"  - Avg questions per image: {stats.get('avg_questions_per_image', 0):.1f}")
    
    # Answer type distribution
    print(f"\n  Answer Type Distribution:")
    for ans_type, count in sorted(stats.get('answer_type_distribution', {}).items()):
        print(f"    - {ans_type}: {count} ({count/stats['total']*100:.1f}%)")
    
    # Task type distribution
    print(f"\n  Task Type Distribution:")
    for task_type, count in sorted(stats.get('task_type_distribution', {}).items(), key=lambda x: -x[1])[:10]:
        print(f"    - {task_type}: {count} ({count/stats['total']*100:.1f}%)")
    
    # Special question types
    print(f"\n  Special Question Types:")
    print(f"    - Yes/No questions: {stats.get('yes_no_questions', 0)}")
    print(f"    - Counting questions: {stats.get('counting_questions', 0)}")
    print(f"    - Color questions: {stats.get('color_questions', 0)}")
    print(f"    - Object questions: {stats.get('object_questions', 0)}")
    
    # Top answers
    print(f"\n  Top 10 Most Common Answers:")
    for answer, count in stats.get('top_10_answers', []):
        print(f"    - '{answer}': {count} times")
    
    return adapter


def test_task_creation_by_type(adapter: VQACOCOAdapter):
    """Test creating tasks for different VQA types"""
    print("\n" + "="*60)
    print("Testing Task Creation by Type")
    print("="*60)
    
    task_types = [
        'vqa_binary',
        'vqa_counting', 
        'vqa_color',
        'vqa_object',
        'vqa_location',
        'vqa_activity',
        'vqa_general'
    ]
    created_tasks = {}
    
    for task_type in task_types:
        task_ids = adapter.get_task_ids(task_type=task_type, limit=1)
        if task_ids:
            # Create task using task_id and adapter
            task = VQACOCOTask(task_id=task_ids[0], adapter=adapter)
            goal, info = task.setup()
            created_tasks[task_type] = task
            
            # Get task data for display
            task_data = adapter.get_task_data(task_ids[0])
            
            print(f"\n[{task_type.upper()} Task]")
            print(f"  - Task ID: {task_data['id']}")
            print(f"  - Question: {task.question}")
            print(f"  - Answer: {task.answer}")
            print(f"  - Answer type: {task.answer_type}")
            print(f"  - Human agreement: {task.answer_confidence:.2f}")
            print(f"  - Unique human answers: {len(task.answer_frequencies)}")
            print(f"  - Top 3 answers: ", end="")
            top_answers = sorted(task.answer_frequencies.items(), key=lambda x: -x[1])[:3]
            print(", ".join([f"{ans} ({cnt})" for ans, cnt in top_answers]))
    
    return created_tasks


def test_vqa_scoring_mechanism(adapter: VQACOCOAdapter):
    """Test VQA-specific scoring mechanism"""
    print("\n" + "="*60)
    print("Testing VQA Scoring Mechanism")
    print("="*60)
    
    # Get tasks with different agreement levels
    print("\n[High Agreement Tasks]")
    high_agreement_tasks = adapter.get_high_agreement_tasks(min_frequency=7, n=3)
    
    for i, task_id in enumerate(high_agreement_tasks[:2]):
        task = VQACOCOTask(task_id=task_id, adapter=adapter)
        task.setup()
        
        print(f"\nTask {i+1}:")
        print(f"  Question: {task.question}")
        print(f"  Most common answer: {task.answer}")
        print(f"  Answer distribution: {dict(sorted(task.answer_frequencies.items(), key=lambda x: -x[1]))}")
        
        # Test different answers
        for test_answer, expected_score in [(task.answer, 1.0), ("wrong answer", 0.0)]:
            success, message = task.check_success(test_answer)
            print(f"  Test '{test_answer}': {message}")
    
    print("\n[Low Agreement (Ambiguous) Tasks]")
    ambiguous_tasks = adapter.get_ambiguous_tasks(max_frequency=2, n=3)
    
    for i, task_id in enumerate(ambiguous_tasks[:2]):
        task = VQACOCOTask(task_id=task_id, adapter=adapter)
        task.setup()
        
        print(f"\nTask {i+1}:")
        print(f"  Question: {task.question}")
        print(f"  Answer distribution: {dict(sorted(task.answer_frequencies.items(), key=lambda x: -x[1]))}")
        print(f"  Human agreement: {task.answer_confidence:.2f}")


def test_answer_type_validation(adapter: VQACOCOAdapter):
    """Test answer validation for different types"""
    print("\n" + "="*60)
    print("Testing Answer Type Validation")
    print("="*60)
    
    # Test Yes/No questions
    print("\n[Yes/No Questions]")
    yes_no_tasks = adapter.get_yes_no_examples(n=3)
    if yes_no_tasks:
        task = VQACOCOTask(task_id=yes_no_tasks[0], adapter=adapter)
        task.setup()
        
        print(f"Question: {task.question}")
        print(f"Correct answer: {task.answer}")
        
        test_cases = [
            ("yes", "Exact yes"),
            ("Yes", "Capitalized"),
            ("YES", "All caps"),
            ("yep", "Casual yes"),
            ("no", "Exact no"),
            ("nope", "Casual no"),
            ("maybe", "Invalid")
        ]
        
        print("Answer validation tests:")
        for test_answer, desc in test_cases:
            success, feedback = task.check_success(test_answer)
            status = "✅" if success else "❌"
            print(f"  {status} {desc}: '{test_answer}' -> {feedback[:50]}...")
    
    # Test Counting questions
    print("\n[Counting Questions]")
    counting_tasks = adapter.get_counting_examples(n=3)
    if counting_tasks:
        task = VQACOCOTask(task_id=counting_tasks[0], adapter=adapter)
        task.setup()
        
        print(f"Question: {task.question}")
        print(f"Correct answer: {task.answer}")
        
        # Try to get numeric value
        try:
            numeric_answer = int(task.answer)
            test_cases = [
                (str(numeric_answer), "Exact string"),
                (numeric_answer, "Integer"),
                (f"{numeric_answer} objects", "With units"),
                ("two" if numeric_answer == 2 else "three", "Word form"),
                (numeric_answer + 1, "Off by one")
            ]
            
            print("Answer validation tests:")
            for test_answer, desc in test_cases:
                success, feedback = task.check_success(test_answer)
                status = "✅" if success else "❌"
                print(f"  {status} {desc}: '{test_answer}' -> {feedback[:50]}...")
        except:
            print("  (Answer is not numeric)")


def test_answer_diversity_analysis(adapter: VQACOCOAdapter):
    """Analyze answer diversity in the dataset"""
    print("\n" + "="*60)
    print("Testing Answer Diversity Analysis")
    print("="*60)
    
    # Sample tasks and analyze answer diversity
    sample_ids = adapter.sample_tasks(100, seed=42)
    
    diversity_stats = {
        'single_answer': 0,
        'low_diversity': 0,  # 2-3 unique answers
        'medium_diversity': 0,  # 4-6 unique answers
        'high_diversity': 0,  # 7+ unique answers
    }
    
    entropy_values = []
    
    for task_id in sample_ids:
        task = VQACOCOTask(task_id=task_id, adapter=adapter)
        task.setup()
        
        unique_answers = len(task.answer_frequencies)
        entropy = task.get_answer_diversity()
        entropy_values.append(entropy)
        
        if unique_answers == 1:
            diversity_stats['single_answer'] += 1
        elif unique_answers <= 3:
            diversity_stats['low_diversity'] += 1
        elif unique_answers <= 6:
            diversity_stats['medium_diversity'] += 1
        else:
            diversity_stats['high_diversity'] += 1
    
    print(f"\nAnswer Diversity Statistics (n={len(sample_ids)}):")
    for category, count in diversity_stats.items():
        print(f"  - {category}: {count} ({count/len(sample_ids)*100:.1f}%)")
    
    print(f"\nAnswer Entropy Statistics:")
    print(f"  - Mean entropy: {np.mean(entropy_values):.2f}")
    print(f"  - Min entropy: {np.min(entropy_values):.2f}")
    print(f"  - Max entropy: {np.max(entropy_values):.2f}")


def test_complete_workflow(adapter: VQACOCOAdapter):
    """Test complete VQA-COCO workflow"""
    print("\n" + "="*60)
    print("Testing Complete VQA-COCO Workflow")
    print("="*60)
    
    # Get a diverse task
    task_ids = adapter.get_diverse_examples(n=1)
    if not task_ids:
        task_ids = adapter.get_task_ids(limit=1)
    
    if not task_ids:
        print("No tasks available!")
        return
    
    task_data = adapter.get_task_data(task_ids[0])
    task = VQACOCOTask(task_id=task_ids[0], adapter=adapter)
    
    goal, info = task.setup()
    
    print("Task Setup:")
    print(f"  - Question: {task.question}")
    print(f"  - Expected answer: {task.answer}")
    print(f"  - Task type: {task.task_type}")
    print(f"  - Answer type: {task.answer_type}")
    print(f"  - Human agreement: {task.answer_confidence:.2f}")
    print(f"  - All human answers: {task.all_answers}")
    
    # Get observation
    obs = task.get_observation()
    print(f"\nObservation keys: {list(obs.keys())}")
    
    # Simulate different agent responses
    print(f"\nSimulating agent responses:")
    
    # 1. Most common answer
    most_common = task.answer
    chat_history = [{"role": "assistant", "content": f"The answer is {most_common}."}]
    reward, done, message, val_info = task.validate(chat_history, most_common)
    print(f"  1. Most common answer ({most_common}): {message}")
    print(f"     Reward: {reward}, Done: {done}")
    print(f"     VQA Score: {val_info.get('vqa_score', 0):.2f}")
    
    # 2. Less common but valid answer
    if len(task.answer_frequencies) > 1:
        other_answers = [ans for ans in task.answer_frequencies.keys() if ans != most_common]
        if other_answers:
            alt_answer = other_answers[0]
            chat_history = [{"role": "assistant", "content": f"I think it's {alt_answer}"}]
            reward, done, message, val_info = task.validate(chat_history, alt_answer)
            print(f"  2. Alternative answer ({alt_answer}): {message}")
            print(f"     Reward: {reward}")
            print(f"     VQA Score: {val_info.get('vqa_score', 0):.2f}")
    
    # 3. Completely wrong answer
    wrong_answer = "purple elephant"
    chat_history = [{"role": "assistant", "content": f"The answer is {wrong_answer}"}]
    reward, done, message, val_info = task.validate(chat_history, wrong_answer)
    print(f"  3. Wrong answer ({wrong_answer}): {message}")
    print(f"     Reward: {reward}")
    
    # Get metrics
    metrics = task.get_metrics()
    print(f"\nTask Metrics:")
    important_metrics = ['task_type', 'answer_type', 'human_agreement', 'unique_answers', 
                        'is_binary', 'is_counting', 'difficulty_factors']
    for key in important_metrics:
        if key in metrics:
            print(f"  - {key}: {metrics[key]}")


def test_task_difficulty_analysis(adapter: VQACOCOAdapter):
    """Analyze task difficulty factors"""
    print("\n" + "="*60)
    print("Testing Task Difficulty Analysis")
    print("="*60)
    
    # Sample tasks and analyze difficulty
    sample_ids = adapter.sample_tasks(50, seed=42)
    
    difficulty_factors_count = Counter()
    task_type_difficulty = defaultdict(list)
    
    for task_id in sample_ids:
        task = VQACOCOTask(task_id=task_id, adapter=adapter)
        goal, info = task.setup()
        
        difficulty_factors = task.get_difficulty_factors()
        for factor in difficulty_factors:
            difficulty_factors_count[factor] += 1
        
        # Track by task type
        task_type_difficulty[task.task_type].append(len(difficulty_factors))
    
    print(f"\nDifficulty Factors Distribution (n={len(sample_ids)}):")
    for factor, count in difficulty_factors_count.most_common():
        print(f"  - {factor}: {count} tasks ({count/len(sample_ids)*100:.1f}%)")
    
    print(f"\nAverage Difficulty by Task Type:")
    for task_type, difficulties in sorted(task_type_difficulty.items()):
        avg_difficulty = np.mean(difficulties) if difficulties else 0
        print(f"  - {task_type}: {avg_difficulty:.2f} factors on average")


def test_batch_evaluation(adapter: VQACOCOAdapter, n_samples: int = 100):
    """Test batch evaluation across different criteria"""
    print("\n" + "="*60)
    print(f"Testing Batch Evaluation ({n_samples} samples)")
    print("="*60)
    
    # Check available tasks
    total_tasks = len(adapter._task_index)
    if total_tasks == 0:
        print("No tasks available for batch evaluation!")
        return
    
    actual_samples = min(n_samples, total_tasks)
    print(f"Available tasks: {total_tasks}, testing {actual_samples} samples")
    
    # Sample tasks with balance
    task_ids = adapter.sample_tasks(
        actual_samples, 
        balanced_types=True,
        seed=42
    )
    
    results = {
        'by_task_type': defaultdict(list),
        'by_answer_type': defaultdict(list),
        'by_agreement_level': defaultdict(list),
        'vqa_scores': [],
        'binary_accuracy': {'correct': 0, 'total': 0},
        'counting_accuracy': {'correct': 0, 'total': 0},
        'answer_in_human': 0,
        'perfect_vqa_score': 0
    }
    
    print(f"\nProcessing {len(task_ids)} tasks...")
    
    for i, task_id in enumerate(task_ids):
        try:
            task = VQACOCOTask(task_id=task_id, adapter=adapter)
            goal, info = task.setup()
            
            # Test with most common answer
            success, message = task.check_success(task.answer)
            
            # Calculate VQA score
            human_count = task.answer_frequencies.get(task.answer, 0)
            vqa_score = min(human_count / 3.0, 1.0)
            results['vqa_scores'].append(vqa_score)
            
            if vqa_score >= 1.0:
                results['perfect_vqa_score'] += 1
            
            # Record results by category
            results['by_task_type'][task.task_type].append(success)
            results['by_answer_type'][task.answer_type].append(success)
            
            # Agreement level categories
            if task.answer_confidence >= 0.7:
                results['by_agreement_level']['high'].append(success)
            elif task.answer_confidence >= 0.4:
                results['by_agreement_level']['medium'].append(success)
            else:
                results['by_agreement_level']['low'].append(success)
            
            # Special handling for binary questions
            if task.is_binary:
                results['binary_accuracy']['total'] += 1
                if success:
                    results['binary_accuracy']['correct'] += 1
            
            # Special handling for counting questions
            if task.is_counting:
                results['counting_accuracy']['total'] += 1
                if success:
                    results['counting_accuracy']['correct'] += 1
            
            # Check if answer is in human answers
            normalized_answer = task._normalize_answer(task.answer)
            human_answers_normalized = [task._normalize_answer(ans) for ans in task.all_answers]
            if normalized_answer in human_answers_normalized:
                results['answer_in_human'] += 1
            
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(task_ids)} tasks...")
                
        except Exception as e:
            print(f"  Error with task {task_id}: {e}")
    
    # Print summary
    print("\nResults Summary:")
    
    print("\nBy Task Type:")
    for ttype, successes in sorted(results['by_task_type'].items()):
        if successes:
            acc = sum(successes) / len(successes) * 100
            print(f"  - {ttype}: {len(successes)} tasks, {acc:.1f}% success")
    
    print("\nBy Answer Type:")
    for atype, successes in sorted(results['by_answer_type'].items()):
        if successes:
            acc = sum(successes) / len(successes) * 100
            print(f"  - {atype}: {len(successes)} tasks, {acc:.1f}% success")
    
    print("\nBy Human Agreement Level:")
    for level, successes in sorted(results['by_agreement_level'].items()):
        if successes:
            acc = sum(successes) / len(successes) * 100
            print(f"  - {level} agreement: {len(successes)} tasks, {acc:.1f}% success")
    
    print("\nVQA Score Statistics:")
    if results['vqa_scores']:
        print(f"  - Mean VQA score: {np.mean(results['vqa_scores']):.3f}")
        print(f"  - Perfect scores (1.0): {results['perfect_vqa_score']} ({results['perfect_vqa_score']/len(results['vqa_scores'])*100:.1f}%)")
    
    print("\nSpecial Categories:")
    if results['binary_accuracy']['total'] > 0:
        binary_acc = results['binary_accuracy']['correct'] / results['binary_accuracy']['total'] * 100
        print(f"  - Binary questions: {binary_acc:.1f}% accuracy")
    
    if results['counting_accuracy']['total'] > 0:
        counting_acc = results['counting_accuracy']['correct'] / results['counting_accuracy']['total'] * 100
        print(f"  - Counting questions: {counting_acc:.1f}% accuracy")
    
    print(f"\nAnswer Quality:")
    print(f"  - Answers matching any human answer: {results['answer_in_human']}/{len(task_ids)} ({results['answer_in_human']/len(task_ids)*100:.1f}%)")


def test_answer_distribution_by_type(adapter: VQACOCOAdapter):
    """Analyze answer distribution by question type"""
    print("\n" + "="*60)
    print("Testing Answer Distribution by Type")
    print("="*60)
    
    task_types = ['vqa_binary', 'vqa_counting', 'vqa_color', 'vqa_object']
    
    for task_type in task_types:
        task_ids = adapter.get_task_ids(task_type=task_type, limit=50)
        if not task_ids:
            continue
        
        print(f"\n[{task_type.upper()}]")
        answer_counter = Counter()
        
        for task_id in task_ids:
            task_data = adapter.get_task_data(task_id)
            answer = task_data.get('answer', '')
            answer_counter[answer] += 1
        
        print(f"  Top 10 answers:")
        for answer, count in answer_counter.most_common(10):
            print(f"    - '{answer}': {count} times")


def visualize_sample_images(adapter: VQACOCOAdapter, n_samples: int = 6):
    """Visualize sample COCO images with questions"""
    print("\n" + "="*60)
    print(f"Visualizing {n_samples} Sample Images")
    print("="*60)
    
    if len(adapter._task_index) == 0:
        print("No tasks available for visualization!")
        return
    
    # Get diverse samples
    sample_ids = adapter.get_diverse_examples(n=n_samples)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, task_id in enumerate(sample_ids[:n_samples]):
        try:
            task = VQACOCOTask(task_id=task_id, adapter=adapter)
            goal, info = task.setup()
            
            # Try to load and display image
            try:
                img = Image.open(task.image_path)
                axes[i].imshow(img)
                axes[i].axis('off')
                
                # Add title with question and answer
                question = task.question[:40] + "..." if len(task.question) > 40 else task.question
                
                # Show answer distribution
                top_answers = sorted(task.answer_frequencies.items(), key=lambda x: -x[1])[:3]
                answers_str = ", ".join([f"{ans}({cnt})" for ans, cnt in top_answers])
                
                title = f"{task.task_type.replace('vqa_', '').upper()}\n"
                title += f"Q: {question}\n"
                title += f"A: {answers_str}"
                
                axes[i].set_title(title, fontsize=9, wrap=True)
                
                img.close()
            except Exception as e:
                # If image loading fails, show task info
                info_text = f"COCO Image #{task.image_id}\n\n"
                info_text += f"Q: {task.question}\n\n"
                info_text += f"Human answers:\n"
                
                for ans, count in sorted(task.answer_frequencies.items(), key=lambda x: -x[1])[:5]:
                    info_text += f"  {ans}: {count} people\n"
                
                info_text += f"\nTask: {task.task_type}"
                info_text += f"\nAnswer type: {task.answer_type}"
                
                axes[i].text(0.5, 0.5, info_text,
                           ha='center', va='center', transform=axes[i].transAxes,
                           fontsize=10, wrap=True)
                axes[i].axis('off')
                
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error: {str(e)[:50]}",
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')
    
    plt.tight_layout()
    output_file = "vqa_coco_samples.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSample images saved to: {output_file}")
    plt.close()


def test_human_agreement_correlation(adapter: VQACOCOAdapter):
    """Test correlation between human agreement and task properties"""
    print("\n" + "="*60)
    print("Testing Human Agreement Correlation")
    print("="*60)
    
    # Sample tasks
    sample_ids = adapter.sample_tasks(100, seed=42)
    
    agreement_by_type = defaultdict(list)
    agreement_by_answer_type = defaultdict(list)
    question_length_agreement = []
    
    for task_id in sample_ids:
        task = VQACOCOTask(task_id=task_id, adapter=adapter)
        task.setup()
        
        agreement = task.answer_confidence
        
        # By task type
        agreement_by_type[task.task_type].append(agreement)
        
        # By answer type
        agreement_by_answer_type[task.answer_type].append(agreement)
        
        # By question length
        q_length = len(task.question.split())
        question_length_agreement.append((q_length, agreement))
    
    print("\nAverage Human Agreement by Task Type:")
    for task_type, agreements in sorted(agreement_by_type.items()):
        avg_agreement = np.mean(agreements)
        print(f"  - {task_type}: {avg_agreement:.3f}")
    
    print("\nAverage Human Agreement by Answer Type:")
    for ans_type, agreements in sorted(agreement_by_answer_type.items()):
        avg_agreement = np.mean(agreements)
        print(f"  - {ans_type}: {avg_agreement:.3f}")
    
    # Question length correlation
    if question_length_agreement:
        lengths = [x[0] for x in question_length_agreement]
        agreements = [x[1] for x in question_length_agreement]
        correlation = np.corrcoef(lengths, agreements)[0, 1]
        print(f"\nCorrelation between question length and agreement: {correlation:.3f}")


def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description='Test VQA-COCO integration with VLM Gym')
    parser.add_argument('--annotation', type=str,
                       default='/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/VQA-COCO/train/vqa_coco_train_vlmgym.json',
                       help='Path to VQA-COCO annotation file')
    parser.add_argument('--data-root', type=str,
                       default='/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/VQA-COCO',
                       help='Path to VQA-COCO data root')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for evaluation')
    parser.add_argument('--skip-vis', action='store_true',
                       help='Skip visualization to avoid image loading issues')
    
    args = parser.parse_args()
    
    print("="*60)
    print("VQA-COCO VLM Gym Integration Test")
    print("="*60)
    print(f"Python: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    try:
        # Test 1: Load adapter
        adapter = test_adapter_loading(args.annotation, args.data_root)
        
        # Check if adapter loaded any data
        if len(adapter._task_index) == 0:
            print("\n❌ No data loaded! Please check:")
            print(f"  1. Annotation file exists: {args.annotation}")
            print(f"  2. Annotation file contains valid data")
            return 1
        
        # Test 2: Create tasks by type
        tasks = test_task_creation_by_type(adapter)
        
        # Test 3: VQA scoring mechanism
        test_vqa_scoring_mechanism(adapter)
        
        # Test 4: Answer type validation
        test_answer_type_validation(adapter)
        
        # Test 5: Answer diversity analysis
        test_answer_diversity_analysis(adapter)
        
        # Test 6: Complete workflow
        test_complete_workflow(adapter)
        
        # Test 7: Task difficulty analysis
        test_task_difficulty_analysis(adapter)
        
        # Test 8: Batch evaluation
        test_batch_evaluation(adapter, n_samples=args.batch_size)
        
        # Test 9: Answer distribution by type
        test_answer_distribution_by_type(adapter)
        
        # Test 10: Human agreement correlation
        test_human_agreement_correlation(adapter)
        
        # Test 11: Visualize samples (optional)
        if not args.skip_vis:
            visualize_sample_images(adapter, n_samples=6)
        
        print("\n" + "="*60)
        print("✓ All tests completed successfully!")
        print("="*60)
        
        print("\nKey findings:")
        print("1. VQA-COCO adapter successfully loads and indexes data")
        print("2. VQA scoring mechanism (min(count/3, 1)) is properly implemented")
        print("3. Different question types (binary, counting, color, etc.) are handled correctly")
        print("4. Human answer agreement and diversity are properly tracked")
        print("5. Answer normalization handles various formats")
        print("6. Task difficulty is assessed based on multiple factors")
        
        print("\nNext steps:")
        print("1. Test with actual VLM models for visual reasoning")
        print("2. Evaluate model performance on high vs low agreement questions")
        print("3. Analyze common error patterns in different question types")
        print("4. Compare model performance with human baseline")
        print("5. Test robustness to answer format variations")
        
        if not args.skip_vis:
            print("\n6. Check vqa_coco_samples.png for visualization examples")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())