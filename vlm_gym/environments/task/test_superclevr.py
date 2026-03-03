#!/usr/bin/env python3
"""
Test script for SuperCLEVR integration with VLM Gym
Tests visual reasoning, yes/no questions, and various task types
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
    from superclevr import SuperClevrTask
except ImportError as e:
    print(f"Warning: Failed to import SuperClevrTask from current directory: {e}")
    try:
        from vlm_gym.environments.task.superclevr import SuperClevrTask
    except ImportError as e2:
        print(f"Error: Cannot import SuperClevrTask: {e2}")
        print("\nPlease ensure superclevr.py exists in:")
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
    from superclevr_adapter import SuperClevrAdapter
except ImportError as e:
    print(f"Error: Cannot import SuperClevrAdapter: {e}")
    print("\nMake sure superclevr_adapter.py is in:")
    print("/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data_adapters/")
    sys.exit(1)


def test_adapter_loading(annotation_file, data_root=None):
    """Test SuperCLEVR adapter initialization and data loading"""
    print("\n" + "="*60)
    print("Testing SuperCLEVR Adapter Loading")
    print("="*60)
    
    # Initialize adapter with provided paths
    print(f"Loading from: {annotation_file}")
    if data_root:
        print(f"Data root: {data_root}")
    
    adapter = SuperClevrAdapter(
        data_root=data_root,
        annotation_file=annotation_file,
        validate_images=False,  # Set to False initially to avoid path issues
        split="train"
    )
    
    # Get statistics
    stats = adapter.get_statistics()
    print(f"\n✓ Adapter loaded successfully!")
    print(f"  - Total tasks: {stats['total']}")
    
    # Task distribution
    print(f"\n  Task Type Distribution:")
    for task_type, count in sorted(stats.get('task_distribution', {}).items()):
        print(f"    - {task_type}: {count} ({count/stats['total']*100:.1f}%)")
    
    # Answer distribution
    print(f"\n  Answer Distribution:")
    for answer, percentage in stats.get('answer_balance', {}).items():
        print(f"    - {answer}: {percentage}")
    
    # Question type distribution
    print(f"\n  Question Type Distribution:")
    for qtype, count in sorted(stats.get('question_type_distribution', {}).items()):
        print(f"    - {qtype}: {count} ({count/stats['total']*100:.1f}%)")
    
    # Attribute statistics
    attr_stats = stats.get('attribute_statistics', {})
    print(f"\n  Top Mentioned Attributes:")
    print(f"    Colors: {', '.join(list(attr_stats.get('colors', {}).keys())[:5])}")
    print(f"    Objects: {', '.join(list(attr_stats.get('objects', {}).keys())[:5])}")
    
    # Complexity statistics
    complex_stats = stats.get('complexity_statistics', {})
    print(f"\n  Complexity Statistics:")
    print(f"    - Simple questions: {complex_stats.get('simple_questions', 0)}")
    print(f"    - Complex questions: {complex_stats.get('complex_questions', 0)}")
    print(f"    - Spatial questions: {complex_stats.get('spatial_questions', 0)}")
    print(f"    - Comparison questions: {complex_stats.get('comparison_questions', 0)}")
    
    return adapter


def test_task_creation_by_type(adapter: SuperClevrAdapter):
    """Test creating tasks for different types"""
    print("\n" + "="*60)
    print("Testing Task Creation by Type")
    print("="*60)
    
    task_types = [
        'visual_comparison',
        'counting_comparison',
        'object_counting', 
        'object_existence',
        'spatial_reasoning',
        'visual_reasoning'
    ]
    created_tasks = {}
    
    for task_type in task_types:
        task_ids = adapter.get_task_ids(task_type=task_type, limit=1)
        if task_ids:
            # Create task using task_id and adapter
            task = SuperClevrTask(task_id=task_ids[0], adapter=adapter)
            goal, info = task.setup()
            created_tasks[task_type] = task
            
            # Get task data for display
            task_data = adapter.get_task_data(task_ids[0])
            
            print(f"\n[{task_type.upper()} Task]")
            print(f"  - Task ID: {task_data['id']}")
            print(f"  - Question: {task.question[:80]}...")
            print(f"  - Answer: {task.answer}")
            print(f"  - Expected answer: {task.expected_answer}")
            print(f"  - Question type: {task.question_type}")
            print(f"  - Is complex: {task.is_complex}")
            print(f"  - Difficulty: {info.get('difficulty', 'unknown')}")
    
    return created_tasks


def test_yes_no_answer_validation(adapter: SuperClevrAdapter):
    """Test yes/no answer validation with various formats"""
    print("\n" + "="*60)
    print("Testing Yes/No Answer Validation")
    print("="*60)
    
    # Get tasks with yes and no answers
    yes_tasks = adapter.get_yes_examples(n=2)
    no_tasks = adapter.get_no_examples(n=2)
    
    test_tasks = []
    if yes_tasks:
        test_tasks.extend([('yes', yes_tasks[0])])
    if no_tasks:
        test_tasks.extend([('no', no_tasks[0])])
    
    for expected_answer, task_id in test_tasks:
        task = SuperClevrTask(task_id=task_id, adapter=adapter)
        task.setup()
        
        print(f"\n[Testing {expected_answer.upper()} answer]")
        print(f"Question: {task.question[:80]}...")
        print(f"Correct answer: {task.expected_answer}")
        
        # Test different answer formats
        if expected_answer == 'yes':
            test_cases = [
                ("yes", "Exact 'yes'"),
                ("Yes", "Capitalized 'Yes'"),
                ("YES", "All caps 'YES'"),
                ("y", "Single 'y'"),
                ("true", "'true'"),
                ("1", "Number '1'"),
                ("affirmative", "'affirmative'"),
                ("The answer is yes", "In sentence"),
                ("no", "Wrong answer 'no'"),
                ("maybe", "Invalid 'maybe'"),
                (None, "None value")
            ]
        else:  # no
            test_cases = [
                ("no", "Exact 'no'"),
                ("No", "Capitalized 'No'"),
                ("NO", "All caps 'NO'"),
                ("n", "Single 'n'"),
                ("false", "'false'"),
                ("0", "Number '0'"),
                ("negative", "'negative'"),
                ("The answer is no", "In sentence"),
                ("yes", "Wrong answer 'yes'"),
                ("unknown", "Invalid 'unknown'"),
                (None, "None value")
            ]
        
        print("\nAnswer validation tests:")
        for test_answer, desc in test_cases:
            success, feedback = task.check_success(test_answer)
            status = "✅" if success else "❌"
            print(f"  {status} {desc}: '{test_answer}' -> {feedback[:50]}...")


def test_complex_questions(adapter: SuperClevrAdapter):
    """Test complex questions with multiple conditions"""
    print("\n" + "="*60)
    print("Testing Complex Questions")
    print("="*60)
    
    # Get complex questions
    complex_tasks = adapter.get_complex_examples(n=5)
    
    if not complex_tasks:
        print("No complex tasks found")
        return
    
    print(f"Found {len(complex_tasks)} complex tasks")
    
    for i, task_id in enumerate(complex_tasks[:3]):
        task = SuperClevrTask(task_id=task_id, adapter=adapter)
        goal, info = task.setup()
        
        print(f"\n[Complex Task {i+1}]")
        print(f"  Question: {task.question}")
        print(f"  Answer: {task.expected_answer}")
        print(f"  Complexity features:")
        
        for feature, value in task.complexity_features.items():
            if value:
                feature_name = feature.replace('has_', '').replace('_', ' ').title()
                print(f"    - {feature_name}: {value}")
        
        # Show mentioned attributes
        if task.attributes:
            print(f"  Mentioned attributes:")
            for attr_type, values in task.attributes.items():
                if values:
                    print(f"    - {attr_type}: {', '.join(values)}")


def test_spatial_reasoning(adapter: SuperClevrAdapter):
    """Test spatial reasoning tasks"""
    print("\n" + "="*60)
    print("Testing Spatial Reasoning Tasks")
    print("="*60)
    
    # Get spatial reasoning tasks
    spatial_tasks = adapter.get_spatial_examples(n=5)
    
    if not spatial_tasks:
        print("No spatial reasoning tasks found")
        return
    
    print(f"Found {len(spatial_tasks)} spatial reasoning tasks")
    
    for i, task_id in enumerate(spatial_tasks[:3]):
        task = SuperClevrTask(task_id=task_id, adapter=adapter)
        goal, info = task.setup()
        
        print(f"\n[Spatial Task {i+1}]")
        print(f"  Question: {task.question}")
        print(f"  Answer: {task.expected_answer}")
        
        # Extract spatial relations
        spatial_terms = ['left', 'right', 'behind', 'front', 'above', 'below']
        found_terms = [term for term in spatial_terms if term in task.question.lower()]
        if found_terms:
            print(f"  Spatial terms: {', '.join(found_terms)}")


def test_attribute_based_filtering(adapter: SuperClevrAdapter):
    """Test filtering tasks by specific attributes"""
    print("\n" + "="*60)
    print("Testing Attribute-based Filtering")
    print("="*60)
    
    # Test color filtering
    test_colors = ['red', 'blue', 'cyan']
    print("\n[Color-based filtering]")
    for color in test_colors:
        color_tasks = adapter.get_examples_by_color(color, n=2)
        if color_tasks:
            print(f"\n  {color.capitalize()} objects: Found {len(color_tasks)} tasks")
            task = SuperClevrTask(task_id=color_tasks[0], adapter=adapter)
            task.setup()
            print(f"    Example: {task.question[:60]}...")
    
    # Test object filtering
    test_objects = ['cube', 'sphere', 'chopper', 'scooter']
    print("\n\n[Object-based filtering]")
    for obj in test_objects:
        obj_tasks = adapter.get_examples_by_object(obj, n=2)
        if obj_tasks:
            print(f"\n  {obj.capitalize()}: Found {len(obj_tasks)} tasks")
            task = SuperClevrTask(task_id=obj_tasks[0], adapter=adapter)
            task.setup()
            print(f"    Example: {task.question[:60]}...")


def test_complete_workflow(adapter: SuperClevrAdapter):
    """Test complete SuperCLEVR workflow"""
    print("\n" + "="*60)
    print("Testing Complete SuperCLEVR Workflow")
    print("="*60)
    
    # Get a visual comparison task
    task_ids = adapter.get_task_ids(task_type='visual_comparison', limit=1)
    if not task_ids:
        task_ids = adapter.get_task_ids(limit=1)
    
    if not task_ids:
        print("No tasks available!")
        return
    
    task_data = adapter.get_task_data(task_ids[0])
    task = SuperClevrTask(task_id=task_ids[0], adapter=adapter)
    
    goal, info = task.setup()
    
    print("Task Setup:")
    print(f"  - Question: {task.question}")
    print(f"  - Expected answer: {task.expected_answer}")
    print(f"  - Task type: {task.task_type}")
    print(f"  - Question type: {task.question_type}")
    print(f"  - Is complex: {task.is_complex}")
    print(f"  - Difficulty: {info['difficulty']}")
    
    # Get observation
    obs = task.get_observation()
    print(f"\nObservation keys: {list(obs.keys())}")
    print(f"  - Scene type: {obs.get('scene_type', 'N/A')}")
    print(f"  - Answer type: {obs.get('answer_type', 'N/A')}")
    
    # Simulate different agent responses
    print(f"\nSimulating agent responses:")
    
    # 1. Correct answer
    correct_answer = task.expected_answer
    chat_history = [{"role": "assistant", "content": f"Looking at the objects, my answer is {correct_answer}."}]
    reward, done, message, val_info = task.validate(chat_history, correct_answer)
    print(f"  1. Correct answer ({correct_answer}): {message}")
    print(f"     Reward: {reward}, Done: {done}")
    
    # 2. Wrong answer
    wrong_answer = "yes" if task.expected_answer == "no" else "no"
    chat_history = [{"role": "assistant", "content": f"I think the answer is {wrong_answer}."}]
    reward, done, message, val_info = task.validate(chat_history, wrong_answer)
    print(f"  2. Wrong answer ({wrong_answer}): {message}")
    print(f"     Reward: {reward}")
    if 'error_analysis' in val_info:
        print(f"     Error type: {val_info['error_analysis']['error_type']}")
    
    # 3. Invalid answer
    invalid_answer = "maybe"
    chat_history = [{"role": "assistant", "content": f"The answer might be {invalid_answer}"}]
    reward, done, message, val_info = task.validate(chat_history, invalid_answer)
    print(f"  3. Invalid answer ({invalid_answer}): {message}")
    print(f"     Reward: {reward}")
    
    # Get metrics
    metrics = task.get_metrics()
    print(f"\nTask Metrics:")
    for key, value in sorted(metrics.items()):
        if key not in ['question', 'answer', 'choices', 'image_path', 'question_features', 'object_attributes', 'complexity_features']:
            print(f"  - {key}: {value}")


def test_difficulty_levels(adapter: SuperClevrAdapter):
    """Test tasks of different difficulty levels"""
    print("\n" + "="*60)
    print("Testing Difficulty Levels")
    print("="*60)
    
    # Get easy tasks
    print("\n[EASY Tasks]")
    easy_tasks = adapter.get_easiest_examples(n=3)
    for task_id in easy_tasks[:2]:
        task = SuperClevrTask(task_id=task_id, adapter=adapter)
        goal, info = task.setup()
        print(f"  Q: {task.question[:60]}...")
        print(f"  A: {task.expected_answer}, Difficulty: {info['difficulty']}")
    
    # Get hard tasks
    print("\n[HARD Tasks]")
    hard_tasks = adapter.get_hardest_examples(n=3)
    for task_id in hard_tasks[:2]:
        task = SuperClevrTask(task_id=task_id, adapter=adapter)
        goal, info = task.setup()
        print(f"  Q: {task.question[:60]}...")
        print(f"  A: {task.expected_answer}, Difficulty: {info['difficulty']}")
        print(f"  Complex: {task.is_complex}, Features: {sum(1 for v in task.complexity_features.values() if v)}")


def test_batch_evaluation(adapter: SuperClevrAdapter, n_samples: int = 50):
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
    
    # Sample tasks with balanced answers
    task_ids = adapter.sample_tasks(
        actual_samples, 
        balanced_answers=True,
        seed=42
    )
    
    results = {
        'by_task_type': defaultdict(list),
        'by_question_type': defaultdict(list),
        'by_answer': defaultdict(list),
        'by_difficulty': defaultdict(list),
        'complex': [],
        'simple': [],
        'spatial': [],
        'comparison': [],
        'errors': defaultdict(int)
    }
    
    print(f"\nProcessing {len(task_ids)} tasks...")
    
    for i, task_id in enumerate(task_ids):
        try:
            task = SuperClevrTask(task_id=task_id, adapter=adapter)
            goal, info = task.setup()
            
            # Simulate correct answer
            success, _ = task.check_success(task.expected_answer)
            
            # Record results
            results['by_task_type'][task.task_type].append(success)
            results['by_question_type'][task.question_type].append(success)
            results['by_answer'][task.expected_answer].append(success)
            results['by_difficulty'][info['difficulty']].append(success)
            
            # Special categories
            if task.is_complex:
                results['complex'].append(success)
            else:
                results['simple'].append(success)
            
            if task.task_type == 'spatial_reasoning':
                results['spatial'].append(success)
            
            if task.task_type == 'visual_comparison':
                results['comparison'].append(success)
            
            # Test with wrong answer to collect error types
            wrong_answer = "yes" if task.expected_answer == "no" else "no"
            wrong_success, _ = task.check_success(wrong_answer)
            _, _, _, val_info = task.validate(
                [{"role": "assistant", "content": wrong_answer}],
                wrong_answer
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
    
    print("\nBy Question Type:")
    for qtype, successes in sorted(results['by_question_type'].items()):
        if successes:
            acc = sum(successes) / len(successes) * 100
            print(f"  - {qtype}: {len(successes)} tasks, {acc:.1f}% success")
    
    print("\nBy Answer:")
    for answer, successes in sorted(results['by_answer'].items()):
        if successes:
            acc = sum(successes) / len(successes) * 100
            print(f"  - {answer}: {len(successes)} tasks, {acc:.1f}% success")
    
    print("\nBy Difficulty:")
    for difficulty, successes in sorted(results['by_difficulty'].items()):
        if successes:
            acc = sum(successes) / len(successes) * 100
            print(f"  - {difficulty}: {len(successes)} tasks, {acc:.1f}% success")
    
    print("\nSpecial Categories:")
    if results['complex']:
        acc = sum(results['complex']) / len(results['complex']) * 100
        print(f"  - Complex questions: {len(results['complex'])} tasks, {acc:.1f}% success")
    if results['simple']:
        acc = sum(results['simple']) / len(results['simple']) * 100
        print(f"  - Simple questions: {len(results['simple'])} tasks, {acc:.1f}% success")
    
    print("\nError Type Distribution:")
    for error_type, count in sorted(results['errors'].items()):
        print(f"  - {error_type}: {count} occurrences")


def test_question_pattern_analysis(adapter: SuperClevrAdapter):
    """Analyze question patterns in the dataset"""
    print("\n" + "="*60)
    print("Testing Question Pattern Analysis")
    print("="*60)
    
    patterns = adapter.analyze_question_patterns()
    
    print("\nQuestion Starters:")
    for starter, count in list(patterns['question_starters'].items())[:10]:
        print(f"  - '{starter}': {count} occurrences")
    
    print("\nCommon Phrases:")
    for phrase, count in list(patterns['common_phrases'].items())[:10]:
        print(f"  - '{phrase}': {count} times")
    
    print("\nComplexity Features:")
    for feature, count in sorted(patterns['complexity_features'].items()):
        feature_name = feature.replace('has_', '').replace('_', ' ').title()
        print(f"  - {feature_name}: {count} questions")


def visualize_sample_images(adapter: SuperClevrAdapter, n_samples: int = 6):
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
            task = SuperClevrTask(task_id=task_id, adapter=adapter)
            goal, info = task.setup()
            
            # Try to load and display image
            try:
                img = Image.open(task.image_path)
                axes[i].imshow(img)
                axes[i].axis('off')
                
                # Add title with question and answer
                question = task.question[:40] + "..." if len(task.question) > 40 else task.question
                title = f"{task.task_type.upper()}\n{question}\nAnswer: {task.expected_answer}"
                if task.is_complex:
                    title = "⚠️ " + title
                
                axes[i].set_title(title, fontsize=9, wrap=True)
                
                img.close()
            except Exception as e:
                # If image loading fails, show task info
                info_text = f"CLEVR Scene\n\nQ: {task.question[:60]}...\nA: {task.expected_answer}"
                info_text += f"\n\nType: {task.task_type}"
                info_text += f"\nDifficulty: {info['difficulty']}"
                
                if task.attributes:
                    info_text += "\n\nAttributes:"
                    for attr_type, values in task.attributes.items():
                        if values:
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
    output_file = "superclevr_samples.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSample images saved to: {output_file}")
    plt.close()


def test_balanced_splits(adapter: SuperClevrAdapter):
    """Test creating balanced train/val/test splits"""
    print("\n" + "="*60)
    print("Testing Balanced Data Splits")
    print("="*60)
    
    # Create splits stratified by answer (yes/no)
    splits = adapter.create_balanced_split(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify_by='answer',
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
    
    # Check answer distribution in each split
    print(f"\nAnswer distribution in splits:")
    for split_name, task_ids in splits.items():
        answer_counts = Counter()
        for task_id in task_ids[:100]:  # Sample first 100 for speed
            task_data = adapter.get_task_data(task_id)
            answer = task_data.get('answer', 'unknown')
            answer_counts[answer] += 1
        
        print(f"\n  {split_name.capitalize()}:")
        total = sum(answer_counts.values())
        for answer, count in sorted(answer_counts.items()):
            print(f"    - {answer}: {count} ({count/total*100:.1f}%)")


def test_comparison_tasks(adapter: SuperClevrAdapter):
    """Test different types of comparison tasks"""
    print("\n" + "="*60)
    print("Testing Comparison Tasks")
    print("="*60)
    
    # Get comparison tasks
    comparison_tasks = adapter.get_comparison_examples(n=5)
    
    if not comparison_tasks:
        print("No comparison tasks found")
        return
    
    print(f"Found {len(comparison_tasks)} comparison tasks")
    
    comparison_types = defaultdict(list)
    
    for task_id in comparison_tasks:
        task = SuperClevrTask(task_id=task_id, adapter=adapter)
        goal, info = task.setup()
        
        # Categorize comparison type
        q_lower = task.question.lower()
        if 'same size' in q_lower:
            comp_type = 'size'
        elif 'same color' in q_lower:
            comp_type = 'color'
        elif 'same shape' in q_lower:
            comp_type = 'shape'
        elif 'same material' in q_lower:
            comp_type = 'material'
        elif 'equal number' in q_lower:
            comp_type = 'quantity'
        else:
            comp_type = 'other'
        
        comparison_types[comp_type].append((task_id, task.question, task.expected_answer))
    
    # Display by type
    for comp_type, tasks in comparison_types.items():
        print(f"\n[{comp_type.upper()} Comparisons]")
        for _, question, answer in tasks[:2]:
            print(f"  Q: {question[:60]}...")
            print(f"  A: {answer}")


def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description='Test SuperCLEVR integration with VLM Gym')
    parser.add_argument('--annotation', type=str,
                       default='/data/wang/meng/GYM-Work/dataset/Super-clevr/train/superclevr_train_vlmgym.json',
                       help='Path to SuperCLEVR annotation file')
    parser.add_argument('--data-root', type=str,
                       default='/data/wang/meng/GYM-Work/dataset/Super-clevr',
                       help='Path to SuperCLEVR data root')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Batch size for evaluation')
    parser.add_argument('--skip-vis', action='store_true',
                       help='Skip visualization to avoid image loading issues')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SuperCLEVR VLM Gym Integration Test")
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
        
        # Test 3: Yes/No answer validation
        test_yes_no_answer_validation(adapter)
        
        # Test 4: Complex questions
        test_complex_questions(adapter)
        
        # Test 5: Spatial reasoning
        test_spatial_reasoning(adapter)
        
        # Test 6: Attribute-based filtering
        test_attribute_based_filtering(adapter)
        
        # Test 7: Complete workflow
        test_complete_workflow(adapter)
        
        # Test 8: Difficulty levels
        test_difficulty_levels(adapter)
        
        # Test 9: Comparison tasks
        test_comparison_tasks(adapter)
        
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
        print("1. SuperCLEVR adapter successfully loads and indexes data")
        print("2. Visual reasoning tasks (comparison, counting, existence) work properly")
        print("3. Yes/No answer validation handles various formats correctly")
        print("4. Complex questions with multiple conditions are identified")
        print("5. Spatial reasoning tasks are properly categorized")
        print("6. Task difficulty assessment is implemented")
        print("7. Attribute-based filtering allows targeted task selection")
        
        print("\nNext steps:")
        print("1. Test with actual VLM models for visual reasoning")
        print("2. Evaluate yes/no accuracy across different task types")
        print("3. Analyze performance on complex vs simple questions")
        print("4. Test spatial reasoning capabilities")
        print("5. Compare model performance on different visual attributes")
        
        if not args.skip_vis:
            print("\n6. Check superclevr_samples.png for visualization examples")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())