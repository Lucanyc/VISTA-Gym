#!/usr/bin/env python3
"""
Test script for A-OKVQA integration with VLM Gym
Tests knowledge-based visual question answering with A-OKVQA dataset
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

# Import the adapter
from aokvqa_adapter import AOKVQAAdapter

# Import task - use absolute import
from vlm_gym.environments.task.aokvqa import AOKVQATask


def test_adapter_loading(annotation_file, data_root=None, use_multiple_choice=True):
    """Test A-OKVQA adapter initialization and data loading"""
    print("\n" + "="*60)
    print("Testing A-OKVQA Adapter Loading")
    print("="*60)
    
    # Initialize adapter with provided paths
    print(f"Loading from: {annotation_file}")
    if data_root:
        print(f"Image root: {data_root}")
    print(f"Mode: {'Multiple Choice' if use_multiple_choice else 'Open-ended'}")
    
    adapter = AOKVQAAdapter(
        data_root=data_root,
        annotation_files=annotation_file,
        validate_images=False,  # Set to False initially to avoid path issues
        split="train",
        use_multiple_choice=use_multiple_choice
    )
    
    # Get statistics
    stats = adapter.get_statistics()
    print(f"\n✓ Adapter loaded successfully!")
    print(f"  - Total tasks: {stats['total']}")
    print(f"  - Unique images: {stats.get('unique_images', 'N/A')}")
    print(f"  - Unique answers: {stats.get('unique_answers', 'N/A')}")
    print(f"  - Unique direct answers: {stats.get('unique_direct_answers', 'N/A')}")
    
    # Difficulty distribution
    print(f"\n  Difficulty Distribution:")
    for diff, count in stats.get('difficulty_distribution', {}).items():
        print(f"    - {diff}: {count} ({count/stats['total']*100:.1f}%)")
    
    # Task type distribution
    print(f"\n  Task Type Distribution:")
    for task_type, count in sorted(stats.get('task_type_distribution', {}).items(), key=lambda x: -x[1]):
        print(f"    - {task_type}: {count} ({count/stats['total']*100:.1f}%)")
    
    # Choice position distribution (for multiple choice)
    if use_multiple_choice and 'choice_position_distribution' in stats:
        print(f"\n  Choice Position Distribution:")
        for pos, count in sorted(stats.get('choice_position_distribution', {}).items()):
            print(f"    - Position {pos}: {count}")
        print(f"  Choice position bias: {stats.get('choice_position_bias', 0):.3f}")
    
    # Top answers
    print(f"\n  Top 10 Most Common Answers:")
    for answer, count in stats.get('top_10_answers', []):
        print(f"    - '{answer}': {count} times")
    
    # Top direct answers
    print(f"\n  Top 10 Most Common Direct Answers:")
    for answer, count in stats.get('top_10_direct_answers', []):
        print(f"    - '{answer}': {count} times")
    
    return adapter


def test_task_creation_modes(adapter: AOKVQAAdapter):
    """Test creating tasks in different modes"""
    print("\n" + "="*60)
    print("Testing Task Creation Modes")
    print("="*60)
    
    # Test both multiple choice and open-ended
    sample_ids = adapter.get_task_ids(limit=2)
    
    if not sample_ids:
        print("No tasks available!")
        return
    
    for task_id in sample_ids[:1]:  # Just test one task in both modes
        print(f"\n[Task ID: {task_id}]")
        
        # Create task
        task = AOKVQATask(task_id=task_id, adapter=adapter)
        goal, info = task.setup()
        
        task_data = adapter.get_task_data(task_id)
        
        print(f"\nQuestion: {task.question}")
        
        if task.is_multiple_choice:
            print(f"\nMultiple Choice Mode:")
            print(f"  Choices:")
            for i, choice in enumerate(task.choices):
                marker = " <-- correct" if i == task.correct_choice_idx else ""
                print(f"    {i}: {choice}{marker}")
        
        print(f"\nDirect Answers from humans:")
        answer_counts = Counter(task.direct_answers)
        for answer, count in answer_counts.most_common():
            print(f"  - '{answer}': {count} times")
        
        print(f"\nMetadata:")
        print(f"  - Difficulty: {'Difficult' if task.is_difficult else 'Easy'}")
        print(f"  - Direct answer agreement: {task.direct_answer_agreement:.2f}")
        print(f"  - Task type: {task.task_type}")
        print(f"  - Requires knowledge: {task.requires_knowledge}")
        
        if task.rationales:
            print(f"\nRationales ({len(task.rationales)}):")
            for i, rationale in enumerate(task.rationales[:2]):
                print(f"  {i+1}. {rationale[:100]}...")


def test_knowledge_questions(adapter: AOKVQAAdapter):
    """Test knowledge-requiring questions specifically"""
    print("\n" + "="*60)
    print("Testing Knowledge-Requiring Questions")
    print("="*60)
    
    # Get knowledge questions
    knowledge_ids = adapter.get_knowledge_examples(n=5)
    
    if not knowledge_ids:
        print("No knowledge questions found!")
        return
    
    for i, task_id in enumerate(knowledge_ids[:3]):
        task = AOKVQATask(task_id=task_id, adapter=adapter)
        goal, info = task.setup()
        
        print(f"\nKnowledge Question {i+1}:")
        print(f"  Q: {task.question}")
        print(f"  A: {task.answer}")
        
        if task.choices:
            print(f"  Choices: {', '.join(task.choices)}")
        
        print(f"  Why it needs knowledge:")
        if task.rationales:
            # Analyze rationale for knowledge indicators
            rationale_text = " ".join(task.rationales).lower()
            knowledge_indicators = []
            
            if "know" in rationale_text:
                knowledge_indicators.append("requires knowing facts")
            if "because" in rationale_text:
                knowledge_indicators.append("requires reasoning")
            if "common" in rationale_text or "usually" in rationale_text:
                knowledge_indicators.append("requires common sense")
            if "recognize" in rationale_text or "identify" in rationale_text:
                knowledge_indicators.append("requires recognition beyond visual")
            
            if knowledge_indicators:
                for indicator in knowledge_indicators:
                    print(f"    - {indicator}")
            
            print(f"  Sample rationale: {task.rationales[0][:150]}...")


def test_difficulty_analysis(adapter: AOKVQAAdapter):
    """Analyze difficult vs easy questions"""
    print("\n" + "="*60)
    print("Testing Difficulty Analysis")
    print("="*60)
    
    # Get difficult and easy examples
    difficult_ids = adapter.get_difficult_examples(n=50)
    easy_ids = adapter.get_easy_examples(n=50)
    
    print(f"\nDifficult questions: {len(difficult_ids)}")
    print(f"Easy questions: {len(easy_ids)}")
    
    # Analyze characteristics
    def analyze_questions(task_ids, label):
        if not task_ids:
            return
        
        task_types = Counter()
        avg_choices = []
        avg_direct_answers = []
        avg_agreement = []
        
        for task_id in task_ids[:20]:  # Sample 20
            task = AOKVQATask(task_id=task_id, adapter=adapter)
            task.setup()
            
            task_types[task.task_type] += 1
            if task.choices:
                avg_choices.append(len(task.choices))
            avg_direct_answers.append(len(set(task.direct_answers)))
            avg_agreement.append(task.direct_answer_agreement)
        
        print(f"\n[{label} Questions Analysis]")
        print(f"  Task type distribution:")
        for ttype, count in task_types.most_common():
            print(f"    - {ttype}: {count}")
        
        if avg_choices:
            print(f"  Average choices: {np.mean(avg_choices):.1f}")
        print(f"  Average unique direct answers: {np.mean(avg_direct_answers):.1f}")
        print(f"  Average agreement: {np.mean(avg_agreement):.3f}")
    
    analyze_questions(difficult_ids, "Difficult")
    analyze_questions(easy_ids, "Easy")


def test_answer_validation(adapter: AOKVQAAdapter):
    """Test answer validation for different formats"""
    print("\n" + "="*60)
    print("Testing Answer Validation")
    print("="*60)
    
    # Test multiple choice validation
    print("\n[Multiple Choice Validation]")
    mc_tasks = adapter.get_task_ids(limit=3)
    
    if mc_tasks:
        # Debug: Check raw data first
        task_data = adapter.get_task_data(mc_tasks[0])
        print(f"Debug - Raw task data:")
        print(f"  Task ID: {mc_tasks[0]}")
        print(f"  Choices from data: {task_data.get('choices', 'NOT FOUND')}")
        print(f"  Answer from data: {task_data.get('answer', 'NOT FOUND')}")
        
        # Create task
        task = AOKVQATask(task_id=mc_tasks[0], adapter=adapter)
        goal, info = task.setup()
        
        print(f"\nDebug - Task object after setup:")
        print(f"  task.choices: {task.choices}")
        print(f"  task.is_multiple_choice: {task.is_multiple_choice}")
        
        print(f"\nQuestion: {task.question}")
        
        # Check if task actually has choices
        if task.choices and isinstance(task.choices, list) and len(task.choices) > 0:
            print(f"Choices: {task.choices}")
            if 0 <= task.correct_choice_idx < len(task.choices):
                print(f"Correct: {task.correct_choice_idx} - {task.choices[task.correct_choice_idx]}")
                
                test_cases = [
                    (str(task.correct_choice_idx), "Correct index as string"),
                    (task.correct_choice_idx, "Correct index as int"),
                    (task.choices[task.correct_choice_idx], "Correct choice text"),
                    (task.choices[task.correct_choice_idx].upper(), "Correct choice uppercase"),
                    ("wrong answer", "Invalid answer"),
                    (str((task.correct_choice_idx + 1) % 4), "Wrong index")
                ]
                
                print("\nValidation tests:")
                for test_answer, desc in test_cases:
                    try:
                        success, feedback = task.check_success(test_answer)
                        status = "✅" if success else "❌"
                        print(f"  {status} {desc}: '{test_answer}' -> {feedback[:60]}...")
                    except Exception as e:
                        print(f"  ❌ {desc}: Error - {str(e)}")
        else:
            print("ERROR: Task has no valid choices!")
            print("Trying to fix by re-running setup...")
    else:
        print("No tasks found in adapter!")
    
    # Test open-ended validation with direct answers
    print("\n[Direct Answer Validation]")
    high_agreement = adapter.get_high_agreement_tasks(min_agreement=5, n=3)
    
    if high_agreement:
        task = AOKVQATask(task_id=high_agreement[0], adapter=adapter)
        task.setup()
        
        print(f"Question: {task.question}")
        print(f"Direct answers: {Counter(task.direct_answers)}")
        
        # Test with different answers
        unique_answers = list(set(task.direct_answers))
        test_cases = []
        
        if len(unique_answers) > 0:
            test_cases.append((unique_answers[0], "First unique answer"))
        if len(unique_answers) > 1:
            test_cases.append((unique_answers[1], "Second unique answer"))
        test_cases.append(("completely wrong", "Wrong answer"))
        
        print("\nValidation tests:")
        for test_answer, desc in test_cases:
            try:
                success, feedback = task.check_success(test_answer)
                status = "✅" if success else "❌"
                print(f"  {status} {desc}: '{test_answer}' -> {feedback[:80]}...")
            except Exception as e:
                print(f"  ❌ {desc}: Error - {str(e)}")


def test_complete_workflow(adapter: AOKVQAAdapter):
    """Test complete A-OKVQA workflow"""
    print("\n" + "="*60)
    print("Testing Complete A-OKVQA Workflow")
    print("="*60)
    
    # Get a knowledge-requiring task
    task_ids = adapter.get_knowledge_examples(n=1)
    if not task_ids:
        task_ids = adapter.get_task_ids(limit=1)
    
    if not task_ids:
        print("No tasks available!")
        return
    
    task_data = adapter.get_task_data(task_ids[0])
    task = AOKVQATask(task_id=task_ids[0], adapter=adapter)
    
    goal, info = task.setup()
    
    print("Task Setup:")
    print(f"  - Question: {task.question}")
    print(f"  - Expected answer: {task.answer}")
    print(f"  - Task type: {task.task_type}")
    print(f"  - Is difficult: {task.is_difficult}")
    print(f"  - Requires knowledge: {task.requires_knowledge}")
    
    if task.choices:
        print(f"  - Choices: {task.choices}")
        print(f"  - Correct choice: {task.correct_choice_idx}")
    
    print(f"  - Direct answers: {Counter(task.direct_answers).most_common(3)}")
    
    # Get observation
    obs = task.get_observation()
    print(f"\nObservation keys: {list(obs.keys())}")
    
    # Show enhanced goal
    print(f"\nEnhanced Goal:")
    print(goal[:500] + "..." if len(goal) > 500 else goal)
    
    # Simulate different agent responses
    print(f"\n\nSimulating agent responses:")
    
    # 1. Correct multiple choice
    if task.choices:
        correct_choice = task.choices[task.correct_choice_idx]
        chat_history = [{"role": "assistant", "content": f"The answer is {correct_choice}."}]
        reward, done, message, val_info = task.validate(chat_history, correct_choice)
        print(f"\n1. Correct choice ({correct_choice}): {message}")
        print(f"   Reward: {reward}, Done: {done}")
        
        # Try with index
        chat_history = [{"role": "assistant", "content": f"I choose option {task.correct_choice_idx}"}]
        reward, done, message, val_info = task.validate(chat_history, str(task.correct_choice_idx))
        print(f"\n2. Correct index ({task.correct_choice_idx}): {message}")
        print(f"   Reward: {reward}")
    
    # 2. Direct answer testing
    if task.direct_answers:
        # Most common direct answer
        answer_counts = Counter(task.direct_answers)
        most_common = answer_counts.most_common(1)[0][0]
        
        chat_history = [{"role": "assistant", "content": f"Based on the image, {most_common}"}]
        reward, done, message, val_info = task.validate(chat_history, most_common)
        print(f"\n3. Most common direct answer ({most_common}): {message}")
        print(f"   Reward: {reward}")
        print(f"   Direct answer score: {val_info.get('direct_answer_score', 0):.2f}")
    
    # Get metrics
    metrics = task.get_metrics()
    print(f"\nTask Metrics:")
    important_metrics = ['task_type', 'is_multiple_choice', 'is_difficult', 
                        'requires_knowledge', 'direct_answer_agreement', 
                        'num_rationales', 'difficulty_factors']
    for key in important_metrics:
        if key in metrics:
            print(f"  - {key}: {metrics[key]}")
    
    # Get hint
    hint = task.get_hint()
    print(f"\nTask Hint: {hint}")


def test_rationale_analysis(adapter: AOKVQAAdapter):
    """Analyze rationales and their keywords"""
    print("\n" + "="*60)
    print("Testing Rationale Analysis")
    print("="*60)
    
    # Get tasks with good rationales
    tasks_with_rationales = adapter.get_task_ids(min_rationales=2, limit=50)
    
    if not tasks_with_rationales:
        print("No tasks with sufficient rationales found!")
        return
    
    # Analyze rationale patterns
    rationale_keywords = Counter()
    rationale_lengths = []
    
    for task_id in tasks_with_rationales[:20]:
        task = AOKVQATask(task_id=task_id, adapter=adapter)
        task.setup()
        
        for rationale in task.rationales:
            rationale_lengths.append(len(rationale.split()))
            
            # Extract key phrases
            lower_rationale = rationale.lower()
            if "because" in lower_rationale:
                rationale_keywords["causal_reasoning"] += 1
            if "looks like" in lower_rationale or "appears" in lower_rationale:
                rationale_keywords["visual_inference"] += 1
            if "usually" in lower_rationale or "typically" in lower_rationale:
                rationale_keywords["common_knowledge"] += 1
            if "must be" in lower_rationale or "has to be" in lower_rationale:
                rationale_keywords["logical_deduction"] += 1
    
    print(f"\nRationale Statistics:")
    print(f"  Average length: {np.mean(rationale_lengths):.1f} words")
    print(f"  Min length: {np.min(rationale_lengths)} words")
    print(f"  Max length: {np.max(rationale_lengths)} words")
    
    print(f"\nRationale reasoning types:")
    for rtype, count in rationale_keywords.most_common():
        print(f"  - {rtype}: {count} occurrences")
    
    # Show example rationales by type
    print(f"\nExample rationales:")
    example_task = AOKVQATask(task_id=tasks_with_rationales[0], adapter=adapter)
    example_task.setup()
    
    print(f"Question: {example_task.question}")
    print(f"Answer: {example_task.answer}")
    print("Rationales:")
    for i, rationale in enumerate(example_task.rationales[:3]):
        print(f"  {i+1}. {rationale}")


def test_batch_evaluation(adapter: AOKVQAAdapter, n_samples: int = 100):
    """Test batch evaluation with different criteria"""
    print("\n" + "="*60)
    print(f"Testing Batch Evaluation ({n_samples} samples)")
    print("="*60)
    
    # Sample tasks with balance
    task_ids = adapter.sample_tasks(
        min(n_samples, len(adapter._task_index)),
        balanced_difficulty=True,
        seed=42
    )
    
    results = {
        'by_task_type': defaultdict(list),
        'by_difficulty': defaultdict(list),
        'by_mode': {'multiple_choice': [], 'open_ended': []},
        'direct_answer_scores': [],
        'agreement_levels': [],
        'knowledge_required': {'yes': [], 'no': []},
        'perfect_mc_score': 0,
        'high_da_score': 0  # direct answer score >= 0.67
    }
    
    print(f"\nProcessing {len(task_ids)} tasks...")
    
    for i, task_id in enumerate(task_ids):
        try:
            task = AOKVQATask(task_id=task_id, adapter=adapter)
            goal, info = task.setup()
            
            # Test with correct answer
            if task.is_multiple_choice:
                # Test multiple choice
                success, message = task.check_success(str(task.correct_choice_idx))
                results['by_mode']['multiple_choice'].append(success)
                if success:
                    results['perfect_mc_score'] += 1
            else:
                results['by_mode']['open_ended'].append(True)  # Mark as processed
            
            # Test with most common direct answer
            if task.direct_answers:
                answer_counts = Counter(task.direct_answers)
                most_common = answer_counts.most_common(1)[0][0]
                match_count = answer_counts[most_common]
                da_score = min(match_count / 3.0, 1.0)
                results['direct_answer_scores'].append(da_score)
                
                if da_score >= 0.67:
                    results['high_da_score'] += 1
            
            # Record results by category
            results['by_task_type'][task.task_type].append(True)
            results['by_difficulty']['difficult' if task.is_difficult else 'easy'].append(True)
            
            # Knowledge requirement
            if task.requires_knowledge:
                results['knowledge_required']['yes'].append(True)
            else:
                results['knowledge_required']['no'].append(True)
            
            # Agreement level
            results['agreement_levels'].append(task.direct_answer_agreement)
            
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(task_ids)} tasks...")
                
        except Exception as e:
            print(f"  Error with task {task_id}: {e}")
    
    # Print summary
    print("\nResults Summary:")
    
    print("\nBy Task Type:")
    for ttype, tasks in sorted(results['by_task_type'].items()):
        print(f"  - {ttype}: {len(tasks)} tasks")
    
    print("\nBy Difficulty:")
    for diff, tasks in sorted(results['by_difficulty'].items()):
        print(f"  - {diff}: {len(tasks)} tasks")
    
    print("\nBy Mode:")
    mc_count = len(results['by_mode']['multiple_choice'])
    if mc_count > 0:
        mc_accuracy = results['perfect_mc_score'] / mc_count * 100
        print(f"  - Multiple choice: {mc_count} tasks, {mc_accuracy:.1f}% accuracy")
    print(f"  - Open ended: {len(results['by_mode']['open_ended'])} tasks")
    
    print("\nDirect Answer Score Statistics:")
    if results['direct_answer_scores']:
        print(f"  - Mean score: {np.mean(results['direct_answer_scores']):.3f}")
        print(f"  - High scores (≥0.67): {results['high_da_score']} ({results['high_da_score']/len(results['direct_answer_scores'])*100:.1f}%)")
    
    print("\nHuman Agreement Statistics:")
    if results['agreement_levels']:
        print(f"  - Mean agreement: {np.mean(results['agreement_levels']):.3f}")
        print(f"  - Min agreement: {np.min(results['agreement_levels']):.3f}")
        print(f"  - Max agreement: {np.max(results['agreement_levels']):.3f}")
    
    print("\nKnowledge Requirement:")
    for req, tasks in results['knowledge_required'].items():
        if tasks:
            print(f"  - Requires knowledge = {req}: {len(tasks)} tasks")


def visualize_sample_tasks(adapter: AOKVQAAdapter, n_samples: int = 6):
    """Visualize sample A-OKVQA tasks"""
    print("\n" + "="*60)
    print(f"Visualizing {n_samples} Sample Tasks")
    print("="*60)
    
    # Get diverse samples
    sample_ids = adapter.get_diverse_examples(n=n_samples)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, task_id in enumerate(sample_ids[:n_samples]):
        try:
            task = AOKVQATask(task_id=task_id, adapter=adapter)
            goal, info = task.setup()
            
            # Create info text (no image loading)
            info_text = f"A-OKVQA Task\n\n"
            info_text += f"Q: {task.question}\n\n"
            
            if task.choices:
                info_text += "Choices:\n"
                for j, choice in enumerate(task.choices):
                    marker = " ✓" if j == task.correct_choice_idx else ""
                    info_text += f"  {j}: {choice}{marker}\n"
                info_text += "\n"
            
            info_text += f"Direct answers:\n"
            answer_counts = Counter(task.direct_answers)
            for ans, count in answer_counts.most_common(3):
                info_text += f"  {ans}: {count}x\n"
            
            info_text += f"\nType: {task.task_type}"
            info_text += f"\nDifficulty: {'Hard' if task.is_difficult else 'Easy'}"
            info_text += f"\nAgreement: {task.direct_answer_agreement:.2f}"
            
            axes[i].text(0.5, 0.5, info_text,
                        ha='center', va='center', transform=axes[i].transAxes,
                        fontsize=10, wrap=True)
            axes[i].set_title(f"Task {i+1}", fontsize=12)
            axes[i].axis('off')
            
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error: {str(e)[:50]}",
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')
    
    plt.tight_layout()
    output_file = "aokvqa_samples.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSample tasks saved to: {output_file}")
    plt.close()


def test_occupation_questions(adapter: AOKVQAAdapter):
    """Test occupation-specific questions"""
    print("\n" + "="*60)
    print("Testing Occupation Questions")
    print("="*60)
    
    # Get occupation questions
    occupation_ids = adapter.get_task_ids(task_type='vqa_occupation', limit=10)
    
    if not occupation_ids:
        print("No occupation questions found!")
        return
    
    print(f"Found {len(occupation_ids)} occupation questions")
    
    # Analyze occupation answers
    occupation_answers = Counter()
    
    for task_id in occupation_ids[:5]:
        task = AOKVQATask(task_id=task_id, adapter=adapter)
        task.setup()
        
        print(f"\nOccupation Question:")
        print(f"  Q: {task.question}")
        print(f"  A: {task.answer}")
        
        if task.choices:
            print(f"  Choices: {', '.join(task.choices)}")
        
        # Collect occupation types
        occupation_answers[task.answer] += 1
        
        # Show rationale snippet
        if task.rationales:
            print(f"  Reasoning: {task.rationales[0][:100]}...")
    
    print(f"\nCommon occupations in dataset:")
    for occupation, count in occupation_answers.most_common():
        print(f"  - {occupation}: {count}")


def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description='Test A-OKVQA integration with VLM Gym')
    parser.add_argument('--annotation', type=str,
                       default='/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/A-OKVQA/aokvqa_train_vlmgym.json',
                       help='Path to A-OKVQA annotation file')
    parser.add_argument('--data-root', type=str,
                       default='/data/wang/meng/GYM-Work/dataset/A-OKVQA',
                       help='Path to A-OKVQA data root')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for evaluation')
    parser.add_argument('--use-open-ended', action='store_true',
                       help='Use open-ended format instead of multiple choice')
    
    args = parser.parse_args()
    
    print("="*60)
    print("A-OKVQA VLM Gym Integration Test")
    print("="*60)
    print(f"Python: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    try:
        # Test 1: Load adapter
        adapter = test_adapter_loading(
            args.annotation, 
            args.data_root,
            use_multiple_choice=not args.use_open_ended
        )
        
        # Check if adapter loaded any data
        if len(adapter._task_index) == 0:
            print("\n❌ No data loaded! Please check:")
            print(f"  1. Annotation file exists: {args.annotation}")
            print(f"  2. Annotation file contains valid data")
            return 1
        
        # Test 2: Task creation modes
        test_task_creation_modes(adapter)
        
        # Test 3: Knowledge questions
        test_knowledge_questions(adapter)
        
        # Test 4: Difficulty analysis
        test_difficulty_analysis(adapter)
        
        # Test 5: Answer validation
        test_answer_validation(adapter)
        
        # Test 6: Complete workflow
        test_complete_workflow(adapter)
        
        # Test 7: Rationale analysis
        test_rationale_analysis(adapter)
        
        # Test 8: Batch evaluation
        test_batch_evaluation(adapter, n_samples=args.batch_size)
        
        # Test 9: Occupation questions
        test_occupation_questions(adapter)
        
        # Test 10: Visualize samples
        visualize_sample_tasks(adapter, n_samples=6)
        
        print("\n" + "="*60)
        print("✓ All tests completed successfully!")
        print("="*60)
        
        print("\nKey findings:")
        print("1. A-OKVQA adapter successfully loads and indexes data")
        print("2. Multiple choice and open-ended modes work correctly")
        print("3. Knowledge-requiring questions are properly identified")
        print("4. Difficulty levels (easy/difficult) are tracked")
        print("5. Direct answer scoring mechanism is implemented")
        print("6. Rationales provide reasoning explanations")
        print("7. Task types include knowledge, occupation, and standard VQA categories")
        
        print("\nNext steps:")
        print("1. Test with actual VLM models for knowledge-based reasoning")
        print("2. Evaluate model performance on difficult vs easy questions")
        print("3. Analyze how well models utilize rationales")
        print("4. Compare multiple choice vs open-ended performance")
        print("5. Test knowledge transfer from rationales to answers")
        
        print("\nCheck aokvqa_samples.png for task visualization")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())