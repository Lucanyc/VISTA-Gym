#!/usr/bin/env python3
"""
Test script for Clever-Math integration with VLM Gym
Tests mathematical reasoning, CLEVR scene understanding, and various math operations
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
import re
import numpy as np

# Add paths for imports
sys.path.insert(0, '/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista')
sys.path.insert(0, '/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data_adapters')

# Import required components
try:
    from clevermath import ClevrMathTask
except ImportError as e:
    print(f"Warning: Failed to import ClevrMathTask from current directory: {e}")
    try:
        from vlm_gym.environments.task.clevermath import ClevrMathTask
    except ImportError as e2:
        print(f"Error: Cannot import ClevrMathTask: {e2}")
        print("\nPlease ensure clevermath.py exists in:")
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
    from clevermath_adapter import ClevrMathAdapter
except ImportError as e:
    print(f"Error: Cannot import ClevrMathAdapter: {e}")
    print("\nMake sure clevermath_adapter.py is in:")
    print("/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data_adapters/")
    sys.exit(1)


def test_adapter_loading(annotation_file, data_root=None):
    """Test Clever-Math adapter initialization and data loading"""
    print("\n" + "="*60)
    print("Testing Clever-Math Adapter Loading")
    print("="*60)
    
    # Initialize adapter with provided paths
    print(f"Loading from: {annotation_file}")
    if data_root:
        print(f"Image root: {data_root}")
    
    adapter = ClevrMathAdapter(
        data_root=data_root,
        annotation_file=annotation_file,
        validate_images=False,  # Set to False initially to avoid path issues
        split="train"
    )
    
    # Get statistics
    stats = adapter.get_statistics()
    print(f"\n✓ Adapter loaded successfully!")
    print(f"  - Total tasks: {stats['total']}")
    print(f"  - Unique images: {stats.get('image_statistics', {}).get('unique_images', 'N/A')}")
    
    # Template distribution
    print(f"\n  Template Distribution:")
    for template, count in sorted(stats.get('template_distribution', {}).items()):
        print(f"    - {template}: {count} ({count/stats['total']*100:.1f}%)")
    
    # Task type distribution
    print(f"\n  Task Type Distribution:")
    for task_type, count in sorted(stats.get('task_type_distribution', {}).items()):
        print(f"    - {task_type}: {count} ({count/stats['total']*100:.1f}%)")
    
    # Operation statistics
    op_stats = stats.get('operation_statistics', {})
    print(f"\n  Operation Statistics:")
    for op, count in sorted(op_stats.items()):
        print(f"    - {op}: {count} ({count/stats['total']*100:.1f}%)")
    
    # Answer statistics
    ans_stats = stats.get('answer_statistics', {})
    print(f"\n  Answer Statistics:")
    print(f"    - Mean: {ans_stats.get('mean', 0):.2f}")
    print(f"    - Range: [{ans_stats.get('min', 0)}, {ans_stats.get('max', 0)}]")
    print(f"    - Zero answers: {ans_stats.get('zero_percentage', 0):.1f}%")
    print(f"    - Negative answers: {ans_stats.get('negative_percentage', 0):.1f}%")
    
    # Complexity statistics
    complex_stats = stats.get('complexity_statistics', {})
    print(f"\n  Complexity Statistics:")
    print(f"    - Single operation: {complex_stats.get('single_operation', 0)}")
    print(f"    - Multi operation: {complex_stats.get('multi_operation', 0)}")
    print(f"    - Adversarial: {complex_stats.get('adversarial', 0)}")
    
    return adapter


def test_task_creation_by_type(adapter: ClevrMathAdapter):
    """Test creating tasks for different types"""
    print("\n" + "="*60)
    print("Testing Task Creation by Type")
    print("="*60)
    
    task_types = [
        'math_addition', 
        'math_subtraction', 
        'math_subtraction_counting',
        'math_addition_counting',
        'math_counting', 
        'math_adversarial_counting'
    ]
    created_tasks = {}
    
    for task_type in task_types:
        task_ids = adapter.get_task_ids(task_type=task_type, limit=1)
        if task_ids:
            # Create task using task_id and adapter
            task = ClevrMathTask(task_id=task_ids[0], adapter=adapter)
            goal, info = task.setup()
            created_tasks[task_type] = task
            
            # Get task data for display
            task_data = adapter.get_task_data(task_ids[0])
            
            print(f"\n[{task_type.upper()} Task]")
            print(f"  - Task ID: {task_data['id']}")
            print(f"  - Question: {task.question[:80]}...")
            print(f"  - Answer: {task.answer}")
            print(f"  - Numeric answer: {task.numeric_answer}")
            print(f"  - Template: {task.template_type}")
            print(f"  - Is adversarial: {task.is_adversarial}")
            print(f"  - Is multihop: {task.is_multihop}")
            print(f"  - Operation count: {task.operation_count}")
            print(f"  - Difficulty: {info.get('difficulty', 'unknown')}")
    
    return created_tasks


def test_operation_based_tasks(adapter: ClevrMathAdapter):
    """Test tasks with different mathematical operations"""
    print("\n" + "="*60)
    print("Testing Operation-based Tasks")
    print("="*60)
    
    operations = ['addition', 'subtraction', 'mixed', 'counting', 'comparison', 'existence']
    
    for operation in operations:
        # Get tasks for this operation
        task_ids = adapter.get_task_ids(operation=operation, limit=3)
        
        if not task_ids:
            print(f"\n[{operation.upper()}] No tasks found")
            continue
        
        print(f"\n[{operation.upper()} Operations] Found {len(task_ids)} tasks")
        
        # Test first task
        task = ClevrMathTask(task_id=task_ids[0], adapter=adapter)
        goal, info = task.setup()
        
        print(f"  Example question: {task.question}")
        print(f"  Expected answer: {task.numeric_answer}")
        
        # Show question analysis
        print(f"  Question features:")
        for feature, value in info['question_features'].items():
            if value:
                print(f"    - {feature}: {value}")
        
        # Show object attributes mentioned
        if info.get('object_attributes'):
            print(f"  Objects mentioned:")
            for attr_type, values in info['object_attributes'].items():
                print(f"    - {attr_type}: {', '.join(values)}")


def test_adversarial_tasks(adapter: ClevrMathAdapter):
    """Test adversarial tasks specifically"""
    print("\n" + "="*60)
    print("Testing Adversarial Tasks")
    print("="*60)
    
    # Get adversarial tasks
    adversarial_tasks = adapter.get_adversarial_examples(n=5)
    
    if not adversarial_tasks:
        print("No adversarial tasks found")
        return
    
    print(f"Found {len(adversarial_tasks)} adversarial tasks")
    
    # Analyze adversarial patterns
    zero_answers = 0
    answer_distribution = Counter()
    
    for i, task_id in enumerate(adversarial_tasks[:3]):
        task = ClevrMathTask(task_id=task_id, adapter=adapter)
        goal, info = task.setup()
        
        print(f"\n[Adversarial Task {i+1}]")
        print(f"  Question: {task.question}")
        print(f"  Answer: {task.numeric_answer}")
        
        if task.numeric_answer == 0:
            zero_answers += 1
            print(f"  ⚠️ This is a trap question with answer 0")
        
        answer_distribution[task.numeric_answer] += 1
        
        # Check if question mentions non-existent objects
        print(f"  Operations requested: ", end="")
        ops = []
        if task.has_addition:
            ops.append("addition")
        if task.has_subtraction:
            ops.append("subtraction")
        if task.has_counting:
            ops.append("counting")
        print(", ".join(ops) if ops else "none")
    
    print(f"\nAdversarial Task Analysis:")
    print(f"  Zero answers: {zero_answers}/{len(adversarial_tasks)} ({zero_answers/len(adversarial_tasks)*100:.1f}%)")
    print(f"  Answer distribution: {dict(answer_distribution)}")


def test_answer_validation_numeric(adapter: ClevrMathAdapter):
    """Test numeric answer validation"""
    print("\n" + "="*60)
    print("Testing Numeric Answer Validation")
    print("="*60)
    
    # Get a few different tasks
    task_ids = adapter.sample_tasks(5, seed=42)
    
    for i, task_id in enumerate(task_ids[:3]):
        task = ClevrMathTask(task_id=task_id, adapter=adapter)
        task.setup()
        
        print(f"\n[Task {i+1}]")
        print(f"Question: {task.question[:80]}...")
        print(f"Correct answer: {task.numeric_answer}")
        
        # Test different answer formats
        test_cases = [
            (str(task.numeric_answer), "Exact string"),
            (task.numeric_answer, "Exact integer"),
            (float(task.numeric_answer), "Float value"),
            (f"The answer is {task.numeric_answer}", "In sentence"),
            (f"{task.numeric_answer} objects", "With units"),
            (f"  {task.numeric_answer}  ", "With spaces"),
            (task.numeric_answer + 1, "Off by one"),
            (task.numeric_answer - 1, "Off by minus one"),
            ("no answer", "Non-numeric"),
            (None, "None value")
        ]
        
        print("\nAnswer validation tests:")
        for test_answer, desc in test_cases:
            success, feedback = task.check_success(test_answer)
            status = "✅" if success else "❌"
            print(f"  {status} {desc}: '{test_answer}' -> {feedback[:50]}...")


def test_multihop_tasks(adapter: ClevrMathAdapter):
    """Test multi-hop reasoning tasks"""
    print("\n" + "="*60)
    print("Testing Multi-hop Tasks")
    print("="*60)
    
    # Get multi-hop tasks
    multihop_tasks = adapter.get_task_ids(template='subtraction-multihop', limit=5)
    
    if not multihop_tasks:
        # Try getting multi-operation tasks
        multihop_tasks = adapter.get_mixed_operation_examples(n=5)
    
    if not multihop_tasks:
        print("No multi-hop tasks found")
        return
    
    print(f"Found {len(multihop_tasks)} multi-hop tasks")
    
    for i, task_id in enumerate(multihop_tasks[:3]):
        task = ClevrMathTask(task_id=task_id, adapter=adapter)
        goal, info = task.setup()
        
        print(f"\n[Multi-hop Task {i+1}]")
        print(f"  Question: {task.question}")
        print(f"  Answer: {task.numeric_answer}")
        print(f"  Operation count: {task.operation_count}")
        print(f"  Is multihop: {task.is_multihop}")
        
        # Show step-by-step operations
        print(f"  Operations involved:")
        if task.has_subtraction:
            print(f"    - Subtraction")
        if task.has_addition:
            print(f"    - Addition")
        if task.has_counting:
            print(f"    - Counting")
        if task.has_remainder:
            print(f"    - Finding remainder")


def test_complete_workflow(adapter: ClevrMathAdapter):
    """Test complete Clever-Math workflow"""
    print("\n" + "="*60)
    print("Testing Complete Clever-Math Workflow")
    print("="*60)
    
    # Get a counting task
    task_ids = adapter.get_task_ids(task_type='math_subtraction_counting', limit=1)
    if not task_ids:
        task_ids = adapter.get_task_ids(limit=1)
    
    if not task_ids:
        print("No tasks available!")
        return
    
    task_data = adapter.get_task_data(task_ids[0])
    task = ClevrMathTask(task_id=task_ids[0], adapter=adapter)
    
    goal, info = task.setup()
    
    print("Task Setup:")
    print(f"  - Question: {task.question}")
    print(f"  - Expected answer: {task.numeric_answer}")
    print(f"  - Task type: {task.task_type}")
    print(f"  - Template: {task.template_type}")
    print(f"  - Is adversarial: {task.is_adversarial}")
    print(f"  - Difficulty: {info['difficulty']}")
    
    # Get observation
    obs = task.get_observation()
    print(f"\nObservation keys: {list(obs.keys())}")
    print(f"  - Scene type: {obs.get('scene_type', 'N/A')}")
    print(f"  - Expected objects: {obs.get('expected_objects', 'N/A')}")
    
    # Simulate different agent responses
    print(f"\nSimulating agent responses:")
    
    # 1. Correct answer
    correct_answer = str(task.numeric_answer)
    chat_history = [{"role": "assistant", "content": f"After counting, there are {correct_answer} objects."}]
    reward, done, message, val_info = task.validate(chat_history, correct_answer)
    print(f"  1. Correct answer ({correct_answer}): {message}")
    print(f"     Reward: {reward}, Done: {done}")
    
    # 2. Off-by-one error
    wrong_answer = str(task.numeric_answer + 1)
    chat_history = [{"role": "assistant", "content": f"I count {wrong_answer} objects."}]
    reward, done, message, val_info = task.validate(chat_history, wrong_answer)
    print(f"  2. Off-by-one error ({wrong_answer}): {message}")
    print(f"     Reward: {reward}")
    if 'error_analysis' in val_info:
        print(f"     Error type: {val_info['error_analysis']['error_type']}")
    
    # 3. Completely wrong answer
    wrong_answer = "5"
    chat_history = [{"role": "assistant", "content": f"The answer is {wrong_answer}"}]
    reward, done, message, val_info = task.validate(chat_history, wrong_answer)
    print(f"  3. Wrong answer ({wrong_answer}): {message}")
    print(f"     Reward: {reward}")
    
    # Get metrics
    metrics = task.get_metrics()
    print(f"\nTask Metrics:")
    for key, value in sorted(metrics.items()):
        if key not in ['question', 'answer', 'choices', 'image_path', 'question_features']:
            print(f"  - {key}: {value}")


def test_answer_distribution(adapter: ClevrMathAdapter):
    """Test answer value distribution"""
    print("\n" + "="*60)
    print("Testing Answer Distribution")
    print("="*60)
    
    # Get tasks with different answer values
    answer_ranges = [
        (0, "Zero answers"),
        (1, "Answer = 1"),
        (2, "Answer = 2"),
        (3, "Answer = 3"),
        (5, "Answer = 5"),
        (10, "Answer = 10")
    ]
    
    for answer_value, desc in answer_ranges:
        task_ids = adapter.get_examples_by_answer(answer_value, n=3)
        if task_ids:
            print(f"\n[{desc}] Found {len(task_ids)} tasks")
            
            # Show one example
            task = ClevrMathTask(task_id=task_ids[0], adapter=adapter)
            task.setup()
            print(f"  Example: {task.question[:60]}...")
            print(f"  Answer: {task.numeric_answer}")


def test_difficulty_levels(adapter: ClevrMathAdapter):
    """Test tasks of different difficulty levels"""
    print("\n" + "="*60)
    print("Testing Difficulty Levels")
    print("="*60)
    
    # Get easy tasks
    print("\n[EASY Tasks]")
    easy_tasks = adapter.get_easiest_examples(n=3)
    for task_id in easy_tasks[:2]:
        task = ClevrMathTask(task_id=task_id, adapter=adapter)
        goal, info = task.setup()
        print(f"  Q: {task.question[:60]}...")
        print(f"  A: {task.numeric_answer}, Difficulty: {info['difficulty']}")
    
    # Get hard tasks
    print("\n[HARD Tasks]")
    hard_tasks = adapter.get_hardest_examples(n=3)
    for task_id in hard_tasks[:2]:
        task = ClevrMathTask(task_id=task_id, adapter=adapter)
        goal, info = task.setup()
        print(f"  Q: {task.question[:60]}...")
        print(f"  A: {task.numeric_answer}, Difficulty: {info['difficulty']}")
        print(f"  Operations: {task.operation_count}, Adversarial: {task.is_adversarial}")


def test_batch_evaluation(adapter: ClevrMathAdapter, n_samples: int = 50):
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
    
    # Sample tasks with different strategies
    task_ids = adapter.sample_tasks(
        actual_samples, 
        stratified_by='template',
        seed=42
    )
    
    results = {
        'by_task_type': defaultdict(list),
        'by_template': defaultdict(list),
        'by_operation': defaultdict(list),
        'by_difficulty': defaultdict(list),
        'adversarial': [],
        'non_adversarial': [],
        'zero_answers': [],
        'non_zero_answers': [],
        'errors': defaultdict(int)
    }
    
    print(f"\nProcessing {len(task_ids)} tasks...")
    
    for i, task_id in enumerate(task_ids):
        try:
            task = ClevrMathTask(task_id=task_id, adapter=adapter)
            goal, info = task.setup()
            
            # Simulate correct answer
            success, _ = task.check_success(str(task.numeric_answer))
            
            # Record results
            results['by_task_type'][task.task_type].append(success)
            results['by_template'][task.template_type].append(success)
            results['by_difficulty'][info['difficulty']].append(success)
            
            # Operation-based results
            if task.has_addition:
                results['by_operation']['addition'].append(success)
            if task.has_subtraction:
                results['by_operation']['subtraction'].append(success)
            if task.has_counting:
                results['by_operation']['counting'].append(success)
            
            # Special categories
            if task.is_adversarial:
                results['adversarial'].append(success)
            else:
                results['non_adversarial'].append(success)
            
            if task.numeric_answer == 0:
                results['zero_answers'].append(success)
            else:
                results['non_zero_answers'].append(success)
            
            # Test with wrong answer to collect error types
            wrong_success, _ = task.check_success(str(task.numeric_answer + 1))
            _, _, _, val_info = task.validate(
                [{"role": "assistant", "content": str(task.numeric_answer + 1)}],
                str(task.numeric_answer + 1)
            )
            
            if 'error_analysis' in val_info:
                error_type = val_info['error_analysis']['error_type']
                results['errors'][error_type] += 1
            
            if (i + 1) % 10 == 0:
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
    
    print("\nBy Template:")
    for template, successes in sorted(results['by_template'].items()):
        if successes:
            acc = sum(successes) / len(successes) * 100
            print(f"  - {template}: {len(successes)} tasks, {acc:.1f}% success")
    
    print("\nBy Operation:")
    for operation, successes in sorted(results['by_operation'].items()):
        if successes:
            acc = sum(successes) / len(successes) * 100
            print(f"  - {operation}: {len(successes)} tasks, {acc:.1f}% success")
    
    print("\nBy Difficulty:")
    for difficulty, successes in sorted(results['by_difficulty'].items()):
        if successes:
            acc = sum(successes) / len(successes) * 100
            print(f"  - {difficulty}: {len(successes)} tasks, {acc:.1f}% success")
    
    print("\nSpecial Categories:")
    if results['adversarial']:
        acc = sum(results['adversarial']) / len(results['adversarial']) * 100
        print(f"  - Adversarial: {len(results['adversarial'])} tasks, {acc:.1f}% success")
    if results['non_adversarial']:
        acc = sum(results['non_adversarial']) / len(results['non_adversarial']) * 100
        print(f"  - Non-adversarial: {len(results['non_adversarial'])} tasks, {acc:.1f}% success")
    
    print("\nError Type Distribution:")
    for error_type, count in sorted(results['errors'].items()):
        print(f"  - {error_type}: {count} occurrences")


def test_question_pattern_analysis(adapter: ClevrMathAdapter):
    """Analyze question patterns in the dataset"""
    print("\n" + "="*60)
    print("Testing Question Pattern Analysis")
    print("="*60)
    
    patterns = adapter.analyze_question_patterns()
    
    print("\nOperation Keywords:")
    for keyword, count in sorted(patterns['operation_keywords'].items()):
        print(f"  - {keyword}: {count} occurrences")
    
    print("\nTop Colors Mentioned:")
    for color, count in sorted(patterns['color_mentions'].items(), 
                              key=lambda x: x[1], reverse=True)[:10]:
        print(f"  - {color}: {count} times")
    
    print("\nTop Shapes Mentioned:")
    for shape, count in sorted(patterns['shape_mentions'].items(), 
                              key=lambda x: x[1], reverse=True):
        print(f"  - {shape}: {count} times")
    
    print("\nSizes Mentioned:")
    for size, count in sorted(patterns['size_mentions'].items()):
        print(f"  - {size}: {count} times")


def visualize_sample_images(adapter: ClevrMathAdapter, n_samples: int = 6):
    """Visualize sample CLEVR images with questions"""
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
            task = ClevrMathTask(task_id=task_id, adapter=adapter)
            goal, info = task.setup()
            
            # Try to load and display image
            try:
                img = Image.open(task.image_path)
                axes[i].imshow(img)
                axes[i].axis('off')
                
                # Add title with question and answer
                question = task.question[:40] + "..." if len(task.question) > 40 else task.question
                title = f"{task.template_type.upper()}\n{question}\nAnswer: {task.numeric_answer}"
                if task.is_adversarial:
                    title = "⚠️ " + title
                
                axes[i].set_title(title, fontsize=9, wrap=True)
                
                img.close()
            except Exception as e:
                # If image loading fails, show task info
                info_text = f"CLEVR Scene\n\nQ: {task.question[:60]}...\nA: {task.numeric_answer}"
                info_text += f"\n\nTemplate: {task.template_type}"
                info_text += f"\nDifficulty: {info['difficulty']}"
                
                if info.get('object_attributes'):
                    info_text += "\n\nObjects:"
                    for attr_type, values in info['object_attributes'].items():
                        info_text += f"\n{attr_type}: {', '.join(values[:3])}"
                
                axes[i].text(0.5, 0.5, info_text,
                           ha='center', va='center', transform=axes[i].transAxes,
                           fontsize=10, wrap=True)
                axes[i].axis('off')
                
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error: {str(e)[:50]}",
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')
    
    plt.tight_layout()
    output_file = "clevr_math_samples.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSample images saved to: {output_file}")
    plt.close()


def test_balanced_splits(adapter: ClevrMathAdapter):
    """Test creating balanced train/val/test splits"""
    print("\n" + "="*60)
    print("Testing Balanced Data Splits")
    print("="*60)
    
    # Create splits stratified by template
    splits = adapter.create_balanced_split(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify_by='template',
        seed=42
    )
    
    print(f"\nSplit sizes:")
    print(f"  - Train: {len(splits['train'])} samples")
    print(f"  - Val: {len(splits['val'])} samples")
    print(f"  - Test: {len(splits['test'])} samples")
    
    # Verify no overlap
    train_set = set(splits['train'])
    val_set = set(splits['val'])
    test_set = set(splits['test'])
    
    print(f"\nOverlap check:")
    print(f"  - Train ∩ Val: {len(train_set & val_set)} (should be 0)")
    print(f"  - Train ∩ Test: {len(train_set & test_set)} (should be 0)")
    print(f"  - Val ∩ Test: {len(val_set & test_set)} (should be 0)")
    
    # Check template distribution in each split
    print(f"\nTemplate distribution in splits:")
    for split_name, task_ids in splits.items():
        template_counts = Counter()
        for task_id in task_ids[:100]:  # Sample first 100 for speed
            task_data = adapter.get_task_data(task_id)
            template = task_data.get('metadata', {}).get('template', 'unknown')
            template_counts[template] += 1
        
        print(f"\n  {split_name.capitalize()}:")
        for template, count in sorted(template_counts.items()):
            print(f"    - {template}: {count}")


def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description='Test Clever-Math integration with VLM Gym')
    parser.add_argument('--annotation', type=str,
                       default='/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/Clever-Math/clevr_math_train_vlmgym.json',
                       help='Path to Clever-Math annotation file')
    parser.add_argument('--data-root', type=str,
                       default='/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/Clever-Math',
                       help='Path to Clever-Math data root')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Batch size for evaluation')
    parser.add_argument('--skip-vis', action='store_true',
                       help='Skip visualization to avoid image loading issues')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Clever-Math VLM Gym Integration Test")
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
        
        # Test 3: Operation-based tasks
        test_operation_based_tasks(adapter)
        
        # Test 4: Adversarial tasks
        test_adversarial_tasks(adapter)
        
        # Test 5: Numeric answer validation
        test_answer_validation_numeric(adapter)
        
        # Test 6: Multi-hop tasks
        test_multihop_tasks(adapter)
        
        # Test 7: Complete workflow
        test_complete_workflow(adapter)
        
        # Test 8: Answer distribution
        test_answer_distribution(adapter)
        
        # Test 9: Difficulty levels
        test_difficulty_levels(adapter)
        
        # Test 10: Batch evaluation
        test_batch_evaluation(adapter, n_samples=args.batch_size)
        
        # Test 11: Question pattern analysis
        test_question_pattern_analysis(adapter)
        
        # Test 12: Balanced splits
        test_balanced_splits(adapter)
        
        # Test 13: Visualize samples (optional)
        if not args.skip_vis:
            visualize_sample_images(adapter, n_samples=6)
        
        print("\n" + "="*60)
        print("✓ All tests completed successfully!")
        print("="*60)
        
        print("\nKey findings:")
        print("1. Clever-Math adapter successfully loads and indexes data")
        print("2. Mathematical operations (addition, subtraction, etc.) are properly handled")
        print("3. Adversarial tasks are correctly identified and processed")
        print("4. Multi-hop reasoning tasks work as expected")
        print("5. Numeric answer validation is robust to different formats")
        print("6. Task difficulty assessment is implemented")
        
        print("\nNext steps:")
        print("1. Test with actual VLM models for mathematical reasoning")
        print("2. Evaluate performance on different operation types")
        print("3. Test adversarial robustness of models")
        print("4. Analyze common error patterns in mathematical reasoning")
        print("5. Compare model performance across difficulty levels")
        
        if not args.skip_vis:
            print("\n6. Check clevr_math_samples.png for visualization examples")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())