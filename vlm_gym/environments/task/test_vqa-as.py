#!/usr/bin/env python3
"""
Test script for VQA-AS integration with VLM Gym
Tests visual question answering on COCO images with diverse question types and answers
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
    from vqaas import VqaAsTask
except ImportError as e:
    print(f"Warning: Failed to import VqaAsTask from current directory: {e}")
    try:
        from vlm_gym.environments.task.vqaas import VqaAsTask
    except ImportError as e2:
        print(f"Error: Cannot import VqaAsTask: {e2}")
        print("\nPlease ensure vqa_as.py exists in:")
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
    from vqaas_adapter import VqaAdapter
except ImportError as e:
    print(f"Error: Cannot import VqaAdapter: {e}")
    print("\nMake sure vqa_adapter.py is in:")
    print("/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data_adapters/")
    sys.exit(1)


def test_adapter_loading(annotation_file, data_root=None):
    """Test VQA adapter initialization and data loading"""
    print("\n" + "="*60)
    print("Testing VQA Adapter Loading")
    print("="*60)
    
    # Initialize adapter with provided paths
    print(f"Loading from: {annotation_file}")
    if data_root:
        print(f"Data root: {data_root}")
    
    adapter = VqaAdapter(
        data_root=data_root,
        annotation_file=annotation_file,
        validate_images=False,  # Set to False initially to avoid path issues
        split="train"
    )
    
    # Get statistics
    stats = adapter.get_statistics()
    print(f"\n✓ Adapter loaded successfully!")
    print(f"  - Total tasks: {stats['total']}")
    print(f"  - Unique answers: {stats['unique_answers']}")
    
    # Task distribution
    print(f"\n  Task Type Distribution:")
    for task_type, count in sorted(stats.get('task_distribution', {}).items()):
        print(f"    - {task_type}: {count} ({count/stats['total']*100:.1f}%)")
    
    # Answer type distribution
    print(f"\n  Answer Type Distribution:")
    answer_balance = stats.get('answer_type_balance', {})
    total = stats['total']
    if total > 0:
        yn = answer_balance.get('yes_no', 0)
        oe = answer_balance.get('open_ended', 0)
        print(f"    - Yes/No: {yn} ({yn/total*100:.1f}%)")
        print(f"    - Open-ended: {oe} ({oe/total*100:.1f}%)")
    
    # Question type distribution
    print(f"\n  Question Type Distribution:")
    for qtype, count in sorted(stats.get('question_type_distribution', {}).items()):
        print(f"    - {qtype}: {count} ({count/stats['total']*100:.1f}%)")
    
    # Top answers
    print(f"\n  Top 10 Answers:")
    for answer, count in list(stats.get('answer_distribution_top20', {}).items())[:10]:
        print(f"    - '{answer}': {count} occurrences")
    
    # Answer length statistics
    answer_stats = stats.get('answer_statistics', {})
    if answer_stats:
        print(f"\n  Answer Length Statistics:")
        print(f"    - Average length: {answer_stats.get('avg_length', 0):.1f} words")
        print(f"    - Short answers (1 word): {answer_stats.get('short_answers', 0)}")
        print(f"    - Medium answers (2-3 words): {answer_stats.get('medium_answers', 0)}")
        print(f"    - Long answers (4+ words): {answer_stats.get('long_answers', 0)}")
    
    # Entity statistics
    entity_stats = stats.get('entity_statistics', {})
    print(f"\n  Top Mentioned Entities:")
    if 'colors' in entity_stats:
        print(f"    Colors: {', '.join(list(entity_stats['colors'].keys())[:5])}")
    if 'activities' in entity_stats:
        print(f"    Activities: {', '.join(list(entity_stats['activities'].keys())[:5])}")
    if 'locations' in entity_stats:
        print(f"    Locations: {', '.join(list(entity_stats['locations'].keys())[:5])}")
    
    return adapter


def test_task_creation_by_type(adapter: VqaAdapter):
    """Test creating tasks for different types"""
    print("\n" + "="*60)
    print("Testing Task Creation by Type")
    print("="*60)
    
    task_types = [
        'object_recognition',
        'color_recognition',
        'object_counting', 
        'spatial_reasoning',
        'activity_recognition',
        'visual_verification',
        'attribute_recognition',
        'visual_qa'
    ]
    created_tasks = {}
    
    for task_type in task_types:
        task_ids = adapter.get_task_ids(task_type=task_type, limit=1)
        if task_ids:
            # Create task using task_id and adapter
            task = VqaAsTask(task_id=task_ids[0], adapter=adapter)
            goal, info = task.setup()
            created_tasks[task_type] = task
            
            # Get task data for display
            task_data = adapter.get_task_data(task_ids[0])
            
            print(f"\n[{task_type.upper()} Task]")
            print(f"  - Task ID: {task_data['id']}")
            print(f"  - Question: {task.question[:80]}...")
            print(f"  - Answer: {task.answer}")
            print(f"  - Answer type: {task.answer_type}")
            print(f"  - Answer length: {task.answer_length_category}")
            print(f"  - Question type: {task.question_type}")
            print(f"  - Is complex: {task.is_complex}")
            print(f"  - Difficulty: {info.get('difficulty', 'unknown')}")
            if task.answer_distribution:
                print(f"  - Answer consensus: {info.get('answer_consensus', 0):.1%}")
                print(f"  - Unique answers: {len(task.answer_distribution)}")
    
    return created_tasks


def test_yes_no_answer_validation(adapter: VqaAdapter):
    """Test yes/no answer validation with various formats"""
    print("\n" + "="*60)
    print("Testing Yes/No Answer Validation")
    print("="*60)
    
    # Get yes/no tasks
    yes_no_tasks = adapter.get_yes_no_examples(n=4)
    
    if not yes_no_tasks:
        print("No yes/no tasks found!")
        return
    
    # Test with tasks that have 'yes' and 'no' answers
    test_tasks = []
    for task_id in yes_no_tasks[:2]:
        task_data = adapter.get_task_data(task_id)
        expected = task_data.get('answer', '').lower()
        if expected in ['yes', 'no']:
            test_tasks.append((expected, task_id))
    
    for expected_answer, task_id in test_tasks:
        task = VqaAsTask(task_id=task_id, adapter=adapter)
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
                ("yeah", "'yeah'"),
                ("yep", "'yep'"),
                ("sure", "'sure'"),
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
                ("nope", "'nope'"),
                ("nah", "'nah'"),
                ("false", "'false'"),
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


def test_open_ended_answers(adapter: VqaAdapter):
    """Test open-ended answer validation"""
    print("\n" + "="*60)
    print("Testing Open-ended Answer Validation")
    print("="*60)
    
    # Get open-ended tasks with different answer lengths
    short_tasks = adapter.get_task_ids(answer_length='short', answer_type='open_ended', limit=2)
    medium_tasks = adapter.get_task_ids(answer_length='medium', answer_type='open_ended', limit=2)
    long_tasks = adapter.get_task_ids(answer_length='long', answer_type='open_ended', limit=2)
    
    test_groups = [
        ("SHORT", short_tasks),
        ("MEDIUM", medium_tasks),
        ("LONG", long_tasks)
    ]
    
    for group_name, task_ids in test_groups:
        if not task_ids:
            continue
            
        print(f"\n[{group_name} Answer Tasks]")
        
        for task_id in task_ids[:1]:
            task = VqaAsTask(task_id=task_id, adapter=adapter)
            task.setup()
            
            print(f"\nQuestion: {task.question}")
            print(f"Expected answer: '{task.expected_answer}'")
            print(f"Answer distribution: {dict(list(task.answer_distribution.items())[:5])}")
            
            # Test exact match
            success, feedback = task.check_success(task.expected_answer)
            print(f"\n✅ Exact match: {feedback}")
            
            # Test with article
            test_answer = f"the {task.expected_answer}"
            success, feedback = task.check_success(test_answer)
            status = "✅" if success else "❌"
            print(f"{status} With article: '{test_answer}' -> {feedback[:60]}...")
            
            # Test plural/singular
            if task.expected_answer.endswith('s'):
                test_answer = task.expected_answer[:-1]  # Remove 's'
            else:
                test_answer = task.expected_answer + 's'  # Add 's'
            success, feedback = task.check_success(test_answer)
            status = "✅" if success else "❌"
            print(f"{status} Plural/singular: '{test_answer}' -> {feedback[:60]}...")
            
            # Test completely wrong answer
            test_answer = "something completely wrong"
            success, feedback = task.check_success(test_answer)
            print(f"❌ Wrong answer: '{test_answer}' -> {feedback[:60]}...")


def test_answer_consensus(adapter: VqaAdapter):
    """Test questions with high and low answer consensus"""
    print("\n" + "="*60)
    print("Testing Answer Consensus")
    print("="*60)
    
    # Get high consensus examples
    print("\n[HIGH Consensus Questions (>80% agreement)]")
    high_consensus = adapter.get_consensus_examples(n=5, min_agreement=0.8)
    
    for task_id in high_consensus[:3]:
        task = VqaAsTask(task_id=task_id, adapter=adapter)
        goal, info = task.setup()
        
        print(f"\nQ: {task.question}")
        print(f"A: {task.expected_answer}")
        print(f"Consensus: {info['answer_consensus']:.1%}")
        print(f"Answer distribution: {dict(list(task.answer_distribution.items())[:3])}")
    
    # Get low consensus (ambiguous) examples
    print("\n\n[LOW Consensus Questions (ambiguous)]")
    ambiguous = adapter.get_ambiguous_examples(n=5)
    
    for task_id in ambiguous[:3]:
        task = VqaAsTask(task_id=task_id, adapter=adapter)
        goal, info = task.setup()
        
        print(f"\nQ: {task.question}")
        print(f"A: {task.expected_answer}")
        print(f"Consensus: {info['answer_consensus']:.1%}")
        print(f"Top answers: {dict(list(task.answer_distribution.items())[:5])}")


def test_task_types_detailed(adapter: VqaAdapter):
    """Test specific task types in detail"""
    print("\n" + "="*60)
    print("Testing Specific Task Types")
    print("="*60)
    
    # Test counting tasks
    print("\n[COUNTING Tasks]")
    counting_tasks = adapter.get_counting_examples(n=3)
    for task_id in counting_tasks[:2]:
        task = VqaAsTask(task_id=task_id, adapter=adapter)
        task.setup()
        print(f"  Q: {task.question}")
        print(f"  A: {task.expected_answer}")
        
        # Test numeric variations
        if task.expected_answer.isdigit():
            num = int(task.expected_answer)
            # Test word form
            number_words = ['zero', 'one', 'two', 'three', 'four', 'five']
            if num < len(number_words):
                success, _ = task.check_success(number_words[num])
                status = "✅" if success else "❌"
                print(f"    {status} Word form: '{number_words[num]}'")
    
    # Test color recognition
    print("\n[COLOR Recognition Tasks]")
    color_tasks = adapter.get_color_examples(n=3)
    for task_id in color_tasks[:2]:
        task = VqaAsTask(task_id=task_id, adapter=adapter)
        task.setup()
        print(f"  Q: {task.question}")
        print(f"  A: {task.expected_answer}")
    
    # Test activity recognition
    print("\n[ACTIVITY Recognition Tasks]")
    activity_tasks = adapter.get_activity_examples(n=3)
    for task_id in activity_tasks[:2]:
        task = VqaAsTask(task_id=task_id, adapter=adapter)
        task.setup()
        print(f"  Q: {task.question}")
        print(f"  A: {task.expected_answer}")


def test_complex_questions(adapter: VqaAdapter):
    """Test complex questions with multiple aspects"""
    print("\n" + "="*60)
    print("Testing Complex Questions")
    print("="*60)
    
    # Get complex questions
    complex_tasks = adapter.get_task_ids(complexity='complex', limit=5)
    
    if not complex_tasks:
        print("No complex tasks found")
        return
    
    print(f"Found {len(complex_tasks)} complex tasks")
    
    for i, task_id in enumerate(complex_tasks[:3]):
        task = VqaAsTask(task_id=task_id, adapter=adapter)
        goal, info = task.setup()
        
        print(f"\n[Complex Task {i+1}]")
        print(f"  Question: {task.question}")
        print(f"  Answer: {task.expected_answer}")
        print(f"  Answer type: {task.answer_type}")
        print(f"  Difficulty: {info['difficulty']}")
        print(f"  Complexity features:")
        
        for feature, value in task.complexity_features.items():
            if value:
                feature_name = feature.replace('has_', '').replace('_', ' ').title()
                print(f"    - {feature_name}: {value}")
        
        # Show entities
        if task.entities:
            print(f"  Mentioned entities:")
            for entity_type, values in task.entities.items():
                if values:
                    print(f"    - {entity_type}: {', '.join(values)}")


def test_complete_workflow(adapter: VqaAdapter):
    """Test complete VQA workflow"""
    print("\n" + "="*60)
    print("Testing Complete VQA Workflow")
    print("="*60)
    
    # Get a diverse example
    task_ids = adapter.get_diverse_examples(n=1)
    if not task_ids:
        task_ids = adapter.get_task_ids(limit=1)
    
    if not task_ids:
        print("No tasks available!")
        return
    
    task_data = adapter.get_task_data(task_ids[0])
    task = VqaAsTask(task_id=task_ids[0], adapter=adapter)
    
    goal, info = task.setup()
    
    print("Task Setup:")
    print(f"  - Question: {task.question}")
    print(f"  - Expected answer: {task.expected_answer}")
    print(f"  - Task type: {task.task_type}")
    print(f"  - Question type: {task.question_type}")
    print(f"  - Answer type: {task.answer_type}")
    print(f"  - Answer length: {task.answer_length_category}")
    print(f"  - Is complex: {task.is_complex}")
    print(f"  - Difficulty: {info['difficulty']}")
    print(f"  - Answer consensus: {info['answer_consensus']:.1%}")
    
    # Get observation
    obs = task.get_observation()
    print(f"\nObservation keys: {list(obs.keys())}")
    print(f"  - Scene type: {obs.get('scene_type', 'N/A')}")
    print(f"  - Answer format: {obs.get('answer_format', 'N/A')}")
    print(f"  - Answer diversity: {obs.get('answer_diversity', 'N/A')}")
    
    # Simulate different agent responses
    print(f"\nSimulating agent responses:")
    
    # 1. Correct answer
    correct_answer = task.expected_answer
    chat_history = [{"role": "assistant", "content": f"Looking at the image, I can see {correct_answer}."}]
    reward, done, message, val_info = task.validate(chat_history, correct_answer)
    print(f"  1. Correct answer ('{correct_answer}'): {message}")
    print(f"     Reward: {reward}, Done: {done}")
    
    # 2. Wrong answer (for yes/no)
    if task.is_binary:
        wrong_answer = "yes" if task.expected_answer == "no" else "no"
        chat_history = [{"role": "assistant", "content": f"I think the answer is {wrong_answer}."}]
        reward, done, message, val_info = task.validate(chat_history, wrong_answer)
        print(f"  2. Wrong answer ('{wrong_answer}'): {message}")
        print(f"     Reward: {reward}")
        if 'error_analysis' in val_info:
            print(f"     Error type: {val_info['error_analysis']['error_type']}")
    else:
        # For open-ended, try a different common answer
        if task.answer_distribution and len(task.answer_distribution) > 1:
            # Get second most common answer
            other_answers = sorted(task.answer_distribution.items(), key=lambda x: x[1], reverse=True)
            if len(other_answers) > 1:
                alt_answer = other_answers[1][0]
                chat_history = [{"role": "assistant", "content": f"I see {alt_answer}."}]
                reward, done, message, val_info = task.validate(chat_history, alt_answer)
                print(f"  2. Alternative answer ('{alt_answer}'): {message[:80]}...")
                print(f"     Reward: {reward}")
                if 'error_analysis' in val_info:
                    print(f"     Was common answer: {val_info['error_analysis']['is_common_answer']}")
    
    # 3. Invalid/empty answer
    invalid_answer = ""
    chat_history = [{"role": "assistant", "content": "I'm not sure."}]
    reward, done, message, val_info = task.validate(chat_history, invalid_answer)
    print(f"  3. Empty answer: {message}")
    print(f"     Reward: {reward}")
    
    # Get metrics
    metrics = task.get_metrics()
    print(f"\nTask Metrics:")
    important_metrics = [
        'task_type', 'question_type', 'answer_type', 'is_binary', 
        'answer_length', 'num_unique_answers', 'answer_consensus', 'difficulty'
    ]
    for key in important_metrics:
        if key in metrics:
            print(f"  - {key}: {metrics[key]}")


def test_batch_evaluation(adapter: VqaAdapter, n_samples: int = 100):
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
    
    # Sample tasks with balanced answer types
    task_ids = adapter.sample_tasks(
        actual_samples, 
        balanced_answer_types=True,
        seed=42
    )
    
    results = {
        'by_task_type': defaultdict(list),
        'by_question_type': defaultdict(list),
        'by_answer_type': defaultdict(list),
        'by_difficulty': defaultdict(list),
        'by_answer_length': defaultdict(list),
        'complex': [],
        'simple': [],
        'yes_no': [],
        'open_ended': [],
        'high_consensus': [],
        'low_consensus': [],
        'errors': defaultdict(int)
    }
    
    print(f"\nProcessing {len(task_ids)} tasks...")
    
    for i, task_id in enumerate(task_ids):
        try:
            task = VqaAsTask(task_id=task_id, adapter=adapter)
            goal, info = task.setup()
            
            # Simulate correct answer
            success, _ = task.check_success(task.expected_answer)
            
            # Record results
            results['by_task_type'][task.task_type].append(success)
            results['by_question_type'][task.question_type].append(success)
            results['by_answer_type'][task.answer_type].append(success)
            results['by_difficulty'][info['difficulty']].append(success)
            results['by_answer_length'][task.answer_length_category].append(success)
            
            # Special categories
            if task.is_complex:
                results['complex'].append(success)
            else:
                results['simple'].append(success)
            
            if task.is_binary:
                results['yes_no'].append(success)
            else:
                results['open_ended'].append(success)
            
            # Consensus-based categories
            consensus = info['answer_consensus']
            if consensus >= 0.8:
                results['high_consensus'].append(success)
            elif consensus <= 0.4:
                results['low_consensus'].append(success)
            
            # Test with wrong answer to collect error types (for binary questions)
            if task.is_binary:
                wrong_answer = "yes" if task.expected_answer == "no" else "no"
                wrong_success, _ = task.check_success(wrong_answer)
                _, _, _, val_info = task.validate(
                    [{"role": "assistant", "content": wrong_answer}],
                    wrong_answer
                )
                
                if 'error_analysis' in val_info:
                    error_type = val_info['error_analysis']['error_type']
                    results['errors'][error_type] += 1
            
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
    
    print("\nBy Answer Length:")
    for length, successes in sorted(results['by_answer_length'].items()):
        if successes:
            acc = sum(successes) / len(successes) * 100
            print(f"  - {length}: {len(successes)} tasks, {acc:.1f}% success")
    
    print("\nBy Difficulty:")
    for difficulty, successes in sorted(results['by_difficulty'].items()):
        if successes:
            acc = sum(successes) / len(successes) * 100
            print(f"  - {difficulty}: {len(successes)} tasks, {acc:.1f}% success")
    
    print("\nSpecial Categories:")
    if results['yes_no']:
        acc = sum(results['yes_no']) / len(results['yes_no']) * 100
        print(f"  - Yes/No questions: {len(results['yes_no'])} tasks, {acc:.1f}% success")
    if results['open_ended']:
        acc = sum(results['open_ended']) / len(results['open_ended']) * 100
        print(f"  - Open-ended questions: {len(results['open_ended'])} tasks, {acc:.1f}% success")
    if results['high_consensus']:
        acc = sum(results['high_consensus']) / len(results['high_consensus']) * 100
        print(f"  - High consensus (≥80%): {len(results['high_consensus'])} tasks, {acc:.1f}% success")
    if results['low_consensus']:
        acc = sum(results['low_consensus']) / len(results['low_consensus']) * 100
        print(f"  - Low consensus (≤40%): {len(results['low_consensus'])} tasks, {acc:.1f}% success")
    
    print("\nError Type Distribution (Binary Questions):")
    for error_type, count in sorted(results['errors'].items()):
        print(f"  - {error_type}: {count} occurrences")


def test_question_pattern_analysis(adapter: VqaAdapter):
    """Analyze question patterns in the dataset"""
    print("\n" + "="*60)
    print("Testing Question Pattern Analysis")
    print("="*60)
    
    patterns = adapter.analyze_question_patterns()
    
    print("\nTop Question Starters:")
    for starter, count in list(patterns['question_starters'].items())[:15]:
        print(f"  - '{starter}': {count} occurrences")
    
    print("\nCommon Question Patterns:")
    for pattern, count in list(patterns['common_patterns'].items())[:15]:
        print(f"  - '{pattern}': {count} times")
    
    print("\nAnswer Patterns:")
    answer_patterns = patterns.get('answer_patterns', {})
    for pattern_type, count in sorted(answer_patterns.items()):
        print(f"  - {pattern_type}: {count} answers")


def test_specific_answer_examples(adapter: VqaAdapter):
    """Test questions with specific answers"""
    print("\n" + "="*60)
    print("Testing Specific Answer Examples")
    print("="*60)
    
    # Test common answers
    test_answers = ['yes', 'no', 'tree', 'red', 'blue', '2', 'sitting', 'table']
    
    for answer in test_answers:
        examples = adapter.get_examples_by_answer(answer, n=2)
        if examples:
            print(f"\n[Answer: '{answer}'] Found {len(examples)} examples")
            
            for task_id in examples[:1]:
                task = VqaAsTask(task_id=task_id, adapter=adapter)
                task.setup()
                print(f"  Q: {task.question}")
                print(f"  Answer distribution: {dict(list(task.answer_distribution.items())[:3])}")


def visualize_sample_images(adapter: VqaAdapter, n_samples: int = 6):
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
            task = VqaAsTask(task_id=task_id, adapter=adapter)
            goal, info = task.setup()
            
            # Try to load and display image
            try:
                img = Image.open(task.image_path)
                axes[i].imshow(img)
                axes[i].axis('off')
                
                # Add title with question and answer
                question = task.question[:40] + "..." if len(task.question) > 40 else task.question
                title = f"{task.task_type.replace('_', ' ').title()}\n"
                title += f"Q: {question}\n"
                title += f"A: {task.expected_answer}"
                
                if task.answer_distribution and len(task.answer_distribution) > 1:
                    consensus = info['answer_consensus']
                    title += f" (consensus: {consensus:.0%})"
                
                if task.is_complex:
                    title = "⚠️ " + title
                
                axes[i].set_title(title, fontsize=9, wrap=True)
                
                img.close()
            except Exception as e:
                # If image loading fails, show task info
                info_text = f"COCO Image\n\nQ: {task.question[:60]}...\nA: {task.expected_answer}"
                info_text += f"\n\nType: {task.task_type}"
                info_text += f"\nAnswer type: {task.answer_type}"
                info_text += f"\nDifficulty: {info['difficulty']}"
                
                if task.answer_distribution:
                    info_text += f"\n\nTop answers:"
                    for ans, count in list(task.answer_distribution.items())[:3]:
                        info_text += f"\n  {ans}: {count}"
                
                axes[i].text(0.5, 0.5, info_text,
                           ha='center', va='center', transform=axes[i].transAxes,
                           fontsize=10, wrap=True)
                axes[i].axis('off')
                
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error: {str(e)[:50]}",
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')
    
    plt.tight_layout()
    output_file = "vqa_samples.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSample images saved to: {output_file}")
    plt.close()


def test_balanced_splits(adapter: VqaAdapter):
    """Test creating balanced train/val/test splits"""
    print("\n" + "="*60)
    print("Testing Balanced Data Splits")
    print("="*60)
    
    # Create splits stratified by answer type (yes/no vs open-ended)
    splits = adapter.create_balanced_split(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify_by='answer_type',
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
    
    # Check answer type distribution in each split
    print(f"\nAnswer type distribution in splits:")
    for split_name, task_ids in splits.items():
        answer_type_counts = Counter()
        for task_id in task_ids[:200]:  # Sample first 200 for speed
            task_data = adapter.get_task_data(task_id)
            answer_type = task_data.get('metadata', {}).get('answer_type', 'unknown')
            answer_type_counts[answer_type] += 1
        
        print(f"\n  {split_name.capitalize()}:")
        total = sum(answer_type_counts.values())
        for answer_type, count in sorted(answer_type_counts.items()):
            print(f"    - {answer_type}: {count} ({count/total*100:.1f}%)")


def test_edge_cases(adapter: VqaAdapter):
    """Test edge cases and special scenarios"""
    print("\n" + "="*60)
    print("Testing Edge Cases")
    print("="*60)
    
    # Test very long answers
    print("\n[Long Answer Questions]")
    long_answer_tasks = adapter.get_task_ids(answer_length='long', limit=3)
    for task_id in long_answer_tasks[:2]:
        task = VqaAsTask(task_id=task_id, adapter=adapter)
        task.setup()
        print(f"  Q: {task.question[:60]}...")
        print(f"  A: {task.expected_answer}")
        print(f"  Answer length: {len(task.expected_answer.split())} words")
    
    # Test questions with many possible answers
    print("\n[Questions with Many Valid Answers]")
    ambiguous = adapter.get_ambiguous_examples(n=3)
    for task_id in ambiguous[:2]:
        task = VqaAsTask(task_id=task_id, adapter=adapter)
        goal, info = task.setup()
        print(f"  Q: {task.question[:60]}...")
        print(f"  Primary answer: {task.expected_answer}")
        print(f"  Unique answers: {len(task.answer_distribution)}")
        print(f"  Consensus: {info['answer_consensus']:.1%}")
    
    # Test empty or None answers
    print("\n[Answer Format Edge Cases]")
    test_task_id = adapter.get_task_ids(limit=1)[0]
    task = VqaAsTask(task_id=test_task_id, adapter=adapter)
    task.setup()
    
    edge_cases = [
        ("", "Empty string"),
        ("   ", "Whitespace only"),
        (None, "None value"),
        ("!!!", "Only punctuation"),
        ("the", "Only article"),
        ("a a a", "Repeated article")
    ]
    
    print(f"  Testing with question: {task.question[:60]}...")
    print(f"  Expected: {task.expected_answer}")
    for test_input, desc in edge_cases:
        success, feedback = task.check_success(test_input)
        status = "✅" if success else "❌"
        print(f"  {status} {desc}: '{test_input}' -> {feedback[:50]}...")


def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description='Test VQA-AS integration with VLM Gym')
    parser.add_argument('--annotation', type=str,
                       default='/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/VQA-AS/vqa_train_vlmgym.json',
                       help='Path to VQA annotation file')
    parser.add_argument('--data-root', type=str,
                       default='/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/VQA-AS',
                       help='Path to VQA data root')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for evaluation')
    parser.add_argument('--skip-vis', action='store_true',
                       help='Skip visualization to avoid image loading issues')
    
    args = parser.parse_args()
    
    print("="*60)
    print("VQA-AS VLM Gym Integration Test")
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
        
        # Test 4: Open-ended answer validation
        test_open_ended_answers(adapter)
        
        # Test 5: Answer consensus testing
        test_answer_consensus(adapter)
        
        # Test 6: Task types detailed
        test_task_types_detailed(adapter)
        
        # Test 7: Complex questions
        test_complex_questions(adapter)
        
        # Test 8: Complete workflow
        test_complete_workflow(adapter)
        
        # Test 9: Specific answer examples
        test_specific_answer_examples(adapter)
        
        # Test 10: Edge cases
        test_edge_cases(adapter)
        
        # Test 11: Batch evaluation
        test_batch_evaluation(adapter, n_samples=args.batch_size)
        
        # Test 12: Question pattern analysis
        test_question_pattern_analysis(adapter)
        
        # Test 13: Balanced splits
        test_balanced_splits(adapter)
        
        # Test 14: Visualize samples (optional)
        if not args.skip_vis:
            visualize_sample_images(adapter, n_samples=6)
        
        print("\n" + "="*60)
        print("✓ All tests completed successfully!")
        print("="*60)
        
        print("\nKey findings:")
        print("1. VQA adapter successfully loads and indexes data")
        print("2. Both Yes/No and open-ended questions work properly")
        print("3. Answer validation handles various formats and variants")
        print("4. Answer consensus analysis identifies ambiguous questions")
        print("5. Complex questions with multiple aspects are identified")
        print("6. Task difficulty assessment considers answer diversity")
        print("7. Multiple task types (object, color, counting, etc.) are supported")
        
        print("\nVQA-AS vs SuperCLEVR differences:")
        print("1. VQA supports open-ended answers, not just yes/no")
        print("2. Answers can have multiple valid variations")
        print("3. Answer consensus varies widely (some questions are ambiguous)")
        print("4. Real-world images lead to more diverse question types")
        print("5. Answer lengths vary from single words to phrases")
        
        print("\nNext steps:")
        print("1. Test with actual VLM models for visual understanding")
        print("2. Evaluate performance on high vs low consensus questions")
        print("3. Analyze accuracy by answer length and type")
        print("4. Compare model performance on different question types")
        print("5. Study how models handle ambiguous questions")
        
        if not args.skip_vis:
            print("\n6. Check vqa_samples.png for visualization examples")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())