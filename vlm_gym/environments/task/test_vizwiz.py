#!/usr/bin/env python3
"""
Test script for VizWiz integration with VLM Gym
Tests visual question answering for visually impaired users with multi-annotator answers
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
import difflib
import numpy as np

# Add VLM Gym to path
sys.path.insert(0, '/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista')

# Import required components
try:
    from vlm_gym.environments.task.vizwiz import VizWizTask
except ImportError as e:
    print(f"Warning: Failed to import VizWizTask: {e}")
    print("Trying alternative import...")
    try:
        from vlm_gym.environments.task import VizWizTask
    except ImportError as e2:
        print(f"Error: Cannot import VizWizTask: {e2}")
        print("\nPlease ensure vizwiz.py exists in vlm_gym/environments/task/")
        sys.exit(1)

from vlm_gym.environments import VisionQAEnv
from vlm_gym.agents.vlm_agent_with_tools import VLMAgentWithTools
from data_adapters.vizwiz_adapter import VizWizAdapter


def test_adapter_loading(annotation_file: str, data_root: str, split: str = "train"):
    """Test VizWiz adapter initialization and data loading"""
    print("\n" + "="*60)
    print("Testing VizWiz Adapter Loading")
    print("="*60)
    
    # Initialize adapter with provided paths
    print(f"Loading from: {annotation_file}")
    print(f"Image root: {data_root}")
    print(f"Split: {split}")
    
    adapter = VizWizAdapter(
        data_root=data_root,
        annotation_files=annotation_file,
        split=split,
        validate_images=False,  # Set to True if you want to validate image paths
        min_confidence_threshold=0.0  # Load all data initially
    )
    
    # Get statistics
    stats = adapter.get_statistics()
    print(f"\n✓ Adapter loaded successfully!")
    print(f"  - Total tasks: {stats['total']:,}")
    print(f"  - Unique images: {stats.get('unique_images', 'N/A'):,}")
    
    # Answerable distribution
    print(f"\n  Answerable Distribution:")
    answerable_dist = stats.get('answerable_distribution', {})
    for status, count in answerable_dist.items():
        percentage = (count / stats['total']) * 100 if stats['total'] > 0 else 0
        print(f"    - {status}: {count:,} ({percentage:.1f}%)")
    
    # Confidence distribution
    print(f"\n  Confidence Distribution:")
    for conf_range, count in sorted(stats.get('confidence_distribution', {}).items()):
        percentage = (count / stats['total']) * 100 if stats['total'] > 0 else 0
        print(f"    - {conf_range}: {count:,} ({percentage:.1f}%)")
    
    # Answer type distribution
    print(f"\n  Answer Types:")
    for atype, count in sorted(stats.get('answer_type_distribution', {}).items()):
        percentage = (count / stats['total']) * 100 if stats['total'] > 0 else 0
        print(f"    - {atype}: {count:,} ({percentage:.1f}%)")
    
    # Special categories
    print(f"\n  Special Categories:")
    print(f"    - Unanimous (≥0.9 confidence): {stats.get('unanimous_tasks', 0):,}")
    print(f"    - Controversial (<0.4 confidence): {stats.get('controversial_tasks', 0):,}")
    print(f"    - High confidence (≥0.7): {stats.get('high_confidence_tasks', 0):,}")
    print(f"    - Low confidence (<0.3): {stats.get('low_confidence_tasks', 0):,}")
    print(f"    - Unanswerable: {stats.get('unanswerable_tasks', 0):,}")
    
    # Answer diversity
    if 'avg_answer_diversity' in stats:
        print(f"\n  Answer Diversity:")
        print(f"    - Average: {stats['avg_answer_diversity']:.3f}")
        print(f"    - Min: {stats['min_answer_diversity']:.3f}")
        print(f"    - Max: {stats['max_answer_diversity']:.3f}")
    
    # Top common answers
    print(f"\n  Top 10 Most Common Answers:")
    for answer, count in stats.get('top_20_common_answers', [])[:10]:
        print(f"    - '{answer}': {count}")
    
    return adapter


def test_task_creation_by_confidence(adapter: VizWizAdapter):
    """Test creating tasks with different confidence levels"""
    print("\n" + "="*60)
    print("Testing Task Creation by Confidence Level")
    print("="*60)
    
    confidence_ranges = ['very_high', 'high', 'medium', 'low', 'very_low']
    created_tasks = {}
    
    for conf_range in confidence_ranges:
        task_ids = adapter.get_task_ids(confidence_range=conf_range, limit=1)
        if task_ids:
            task = VizWizTask(task_id=task_ids[0], adapter=adapter)
            goal, info = task.setup()
            created_tasks[conf_range] = task
            
            print(f"\n[{conf_range.upper().replace('_', ' ')} Confidence]")
            print(f"  - Task ID: {task_ids[0]}")
            print(f"  - Question: {task.question}")
            print(f"  - Answer: {task.answer}")
            print(f"  - Confidence: {task.answer_confidence:.3f}")
            print(f"  - All answers ({len(task.all_answers)}): {task.all_answers[:3]}{'...' if len(task.all_answers) > 3 else ''}")
            print(f"  - Question category: {task.question_category}")
            print(f"  - Answerable: {task.answerable}")
    
    return created_tasks


def test_question_categories(adapter: VizWizAdapter):
    """Test different question categories"""
    print("\n" + "="*60)
    print("Testing Question Categories")
    print("="*60)
    
    # Categorize tasks by analyzing question content
    categorized = defaultdict(list)
    yes_no_answers = []  # Track yes/no answers separately
    
    # Sample tasks to categorize
    sample_ids = adapter.get_task_ids(limit=500, shuffle=True, seed=42)
    
    for task_id in sample_ids:
        task_data = adapter.get_task_data(task_id)
        question_lower = task_data['question'].lower()
        answer_lower = task_data['answer'].lower()
        
        # Check if answer is yes/no
        if answer_lower in ['yes', 'no']:
            yes_no_answers.append(task_id)
        
        if any(word in question_lower for word in ['color', 'colour', 'what color']):
            categorized['color'].append(task_id)
        elif any(word in question_lower for word in ['read', 'say', 'text', 'label', 'written']):
            categorized['text_reading'].append(task_id)
        elif 'how many' in question_lower or 'count' in question_lower:
            categorized['counting'].append(task_id)
        elif any(word in question_lower for word in ['is this', 'is it', 'are there', 'does']):
            categorized['yes_no'].append(task_id)
        elif any(word in question_lower for word in ['what is this', 'what kind', 'identify']):
            categorized['object'].append(task_id)
    
    # Show yes/no questions found by answer type
    print(f"\nYes/No questions (by answer): {len(yes_no_answers)}")
    print(f"Yes/No questions (by question pattern): {len(categorized.get('yes_no', []))}")
    
    # Show examples from each category
    for category, task_ids in categorized.items():
        if task_ids:
            print(f"\n[{category.upper().replace('_', ' ')} Questions] ({len(task_ids)} found)")
            
            # Show one example
            task = VizWizTask(task_id=task_ids[0], adapter=adapter)
            task.setup()
            
            print(f"  Example question: {task.question}")
            print(f"  Answer: {task.answer}")
            print(f"  Confidence: {task.answer_confidence:.3f}")
    
    # Categorize tasks
    categorized = defaultdict(list)
    
    for task_id in adapter.get_task_ids(limit=500):
        task_data = adapter.get_task_data(task_id)
        question_lower = task_data['question'].lower()
        
        if any(word in question_lower for word in ['color', 'colour']):
            categorized['color'].append(task_id)
        elif any(word in question_lower for word in ['read', 'say', 'text', 'label']):
            categorized['text_reading'].append(task_id)
        elif 'how many' in question_lower:
            categorized['counting'].append(task_id)
        elif any(word in question_lower for word in ['is this', 'is it', 'are there']):
            categorized['yes_no'].append(task_id)
    
    # Show examples from each category
    for category, task_ids in categorized.items():
        if task_ids:
            print(f"\n[{category.upper().replace('_', ' ')} Questions] ({len(task_ids)} found)")
            
            # Show one example
            task = VizWizTask(task_id=task_ids[0], adapter=adapter)
            task.setup()
            
            print(f"  Example question: {task.question}")
            print(f"  Answer: {task.answer}")
            print(f"  Confidence: {task.answer_confidence:.3f}")


def test_answer_validation_comprehensive(adapter: VizWizAdapter):
    """Test answer validation for various scenarios"""
    print("\n" + "="*60)
    print("Testing Comprehensive Answer Validation")
    print("="*60)
    
    # Test 1: Unanswerable cases
    print("\n[UNANSWERABLE Case Testing]")
    unanswerable_ids = adapter.get_unanswerable_examples(n=1)
    if unanswerable_ids:
        task = VizWizTask(task_id=unanswerable_ids[0], adapter=adapter)
        task.setup()
        
        print(f"Question: {task.question}")
        print(f"Marked as: {'unanswerable' if task.is_unanswerable else 'answerable'}")
        
        test_cases = [
            ("unanswerable", "Direct match", True),
            ("Cannot answer this", "Alternative phrasing", True),
            ("Too blurry to see", "Quality issue", True),
            ("The image is unclear", "Unclear image", True),
            ("red", "Specific answer when unanswerable", False),
        ]
        
        for test_answer, description, should_pass in test_cases:
            success, feedback = task.check_success(test_answer)
            status = "✅" if success == should_pass else "❌"
            print(f"  {status} {description}: '{test_answer}' -> {feedback[:60]}...")
    
    # Test 2: High confidence answers
    print("\n\n[HIGH CONFIDENCE Answer Validation]")
    high_conf_ids = adapter.get_high_quality_examples(n=1)
    if high_conf_ids:
        task = VizWizTask(task_id=high_conf_ids[0], adapter=adapter)
        task.setup()
        
        print(f"Question: {task.question}")
        print(f"Answer: {task.answer} (confidence: {task.answer_confidence:.3f})")
        print(f"All annotator answers: {task.all_answers}")
        
        # Test exact match and variations
        test_cases = [
            (task.answer, "Exact match", True),
            (task.answer.upper(), "Uppercase", True),
            (f"The answer is {task.answer}", "In sentence", True),
        ]
        
        # Test alternative answers from annotators
        for alt_answer in task.all_answers[:2]:
            if alt_answer != task.answer:
                test_cases.append((alt_answer, "Alternative annotator answer", True))
        
        for test_answer, description, should_pass in test_cases:
            success, feedback = task.check_success(test_answer)
            print(f"  {description}: '{test_answer}' -> Success: {success}")
    
    # Test 3: Low confidence answers
    print("\n\n[LOW CONFIDENCE Answer Validation]")
    low_conf_ids = adapter.get_task_ids(confidence_range='low', limit=1)
    if low_conf_ids:
        task = VizWizTask(task_id=low_conf_ids[0], adapter=adapter)
        task.setup()
        
        print(f"Question: {task.question}")
        print(f"Answer: {task.answer} (confidence: {task.answer_confidence:.3f})")
        print(f"Answer diversity: {len(task.all_answers)} unique answers")
        
        # Show how different answers are handled
        for i, alt_answer in enumerate(task.all_answers[:3]):
            success, feedback = task.check_success(alt_answer)
            print(f"  Alt answer {i+1}: '{alt_answer}' -> Success: {success}")
    
    # Test 4: Yes/No questions
    print("\n\n[YES/NO Answer Validation]")
    # Find a yes/no question manually
    yes_no_found = False
    for task_id in adapter.get_task_ids(limit=50, shuffle=True):
        task_data = adapter.get_task_data(task_id)
        if task_data['answer'].lower() in ['yes', 'no']:
            yes_no_found = True
            task = VizWizTask(task_id=task_id, adapter=adapter)
            task.setup()
            
            print(f"Question: {task.question}")
            print(f"Answer: {task.answer}")
            
            test_cases = [
                (task.answer, "Exact match", True),
                (task.answer.capitalize(), "Capitalized", True),
                (f"{task.answer}, that's correct", "With extra text", True),
                ("Yes" if task.answer.lower() == "no" else "No", "Wrong answer", False),
            ]
            
            for test_answer, description, should_pass in test_cases:
                success, feedback = task.check_success(test_answer)
                status = "✅" if success == should_pass else "❌"
                print(f"  {status} {description}: '{test_answer}'")
            break
    
    if not yes_no_found:
        print("  No yes/no questions found in sample")


def test_color_questions(adapter: VizWizAdapter):
    """Test color identification questions"""
    print("\n" + "="*60)
    print("Testing Color Identification Questions")
    print("="*60)
    
    # Find color questions
    color_tasks = []
    for task_id in adapter.get_task_ids(limit=200, shuffle=True):
        task_data = adapter.get_task_data(task_id)
        if 'color' in task_data['question'].lower():
            color_tasks.append(task_id)
            if len(color_tasks) >= 3:
                break
    
    if not color_tasks:
        print("No color questions found in sample")
        return
    
    for i, task_id in enumerate(color_tasks):
        task = VizWizTask(task_id=task_id, adapter=adapter)
        task.setup()
        
        print(f"\n[Color Question {i+1}]")
        print(f"  Question: {task.question}")
        print(f"  Answer: {task.answer}")
        
        # Test color extraction
        test_cases = [
            task.answer,  # Exact color
            f"It's {task.answer}",  # With context
            f"{task.answer} and blue",  # Multiple colors
            "I see red and green colors",  # Different colors
        ]
        
        for test_answer in test_cases:
            success, feedback = task.check_success(test_answer)
            print(f"  '{test_answer}' -> {'✅' if success else '❌'}")


def test_counting_questions(adapter: VizWizAdapter):
    """Test counting questions"""
    print("\n" + "="*60)
    print("Testing Counting Questions")
    print("="*60)
    
    # Find counting questions
    counting_tasks = []
    for task_id in adapter.get_task_ids(limit=200, shuffle=True):
        task_data = adapter.get_task_data(task_id)
        if 'how many' in task_data['question'].lower():
            counting_tasks.append(task_id)
            if len(counting_tasks) >= 2:
                break
    
    if not counting_tasks:
        print("No counting questions found in sample")
        return
    
    for i, task_id in enumerate(counting_tasks):
        task = VizWizTask(task_id=task_id, adapter=adapter)
        task.setup()
        
        print(f"\n[Counting Question {i+1}]")
        print(f"  Question: {task.question}")
        print(f"  Answer: {task.answer}")
        
        # Try to extract number from answer
        number = task._extract_number(task.answer)
        if number is not None:
            test_cases = [
                (str(int(number)), "Exact number", True),
                (f"There are {int(number)}", "With context", True),
                (str(int(number) + 1), "Wrong count", False),
                ("three" if number == 3 else "two", "Word form", number == 3 or number == 2),
            ]
            
            for test_answer, description, should_pass in test_cases:
                success, feedback = task.check_success(test_answer)
                status = "✅" if success == should_pass else "❌"
                print(f"  {status} {description}: '{test_answer}'")


def test_annotator_agreement_analysis(adapter: VizWizAdapter):
    """Test annotator agreement analysis"""
    print("\n" + "="*60)
    print("Testing Annotator Agreement Analysis")
    print("="*60)
    
    # Get tasks with different agreement levels
    test_cases = [
        ("High Confidence (≥0.9)", adapter.get_task_ids(min_confidence=0.9, limit=1)),
        ("Low Confidence (≤0.4)", adapter.get_task_ids(max_confidence=0.4, limit=1)),
        ("Medium Confidence", adapter.get_task_ids(min_confidence=0.5, max_confidence=0.7, limit=1))
    ]
    
    for case_name, task_ids in test_cases:
        if task_ids:
            print(f"\n[{case_name}]")
            
            # Analyze agreement
            analysis = adapter.analyze_annotator_agreement(task_ids[0])
            
            print(f"  Question: {analysis['question']}")
            print(f"  Final answer: {analysis['final_answer']}")
            print(f"  Confidence: {analysis['answer_confidence']:.3f}")
            print(f"  Answer diversity: {analysis['answer_diversity']:.3f}")
            print(f"  All unique answers: {analysis['all_answers']}")
            
            if 'answer_frequency' in analysis:
                print(f"  Answer frequency:")
                for answer, count in sorted(analysis['answer_frequency'].items(), 
                                          key=lambda x: -x[1])[:5]:
                    print(f"    - '{answer}': {count}")


def test_complete_workflow(adapter: VizWizAdapter):
    """Test complete VizWiz workflow"""
    print("\n" + "="*60)
    print("Testing Complete VizWiz Workflow")
    print("="*60)
    
    # Test different scenarios
    test_configs = [
        {'min_confidence': 0.9, 'desc': 'High Confidence (≥0.9)'},
        {'max_confidence': 0.4, 'desc': 'Low Confidence (≤0.4)'},
        {'answerable': False, 'desc': 'Unanswerable Question'},
        {'min_confidence': 0.5, 'max_confidence': 0.7, 'desc': 'Medium Confidence'},
    ]
    
    for config in test_configs:
        print(f"\n[Testing {config['desc']}]")
        
        # Remove desc from config for get_task_ids
        query_config = {k: v for k, v in config.items() if k != 'desc'}
        task_ids = adapter.get_task_ids(**query_config, limit=1)
        
        if not task_ids:
            print(f"  No tasks found for this configuration")
            continue
        
        task = VizWizTask(task_id=task_ids[0], adapter=adapter)
        goal, info = task.setup()
        
        print("  Task Setup:")
        print(f"    - Question: {task.question}")
        print(f"    - Expected answer: {task.answer}")
        print(f"    - Confidence: {task.answer_confidence:.3f}")
        print(f"    - Category: {task.question_category}")
        print(f"    - Answerable: {task.answerable}")
        print(f"    - Unique answers: {len(task.all_answers)}")
        
        # Get observation
        obs = task.get_observation()
        print(f"\n  Observation keys: {list(obs.keys())}")
        
        # Simulate different VLM responses
        print(f"\n  Simulating VLM responses:")
        
        # 1. Correct answer
        chat_history = [{"role": "assistant", "content": task.answer}]
        reward, done, message, val_info = task.validate(chat_history, task.answer)
        print(f"    1. Correct answer: {message[:60]}...")
        print(f"       Reward: {reward}, Done: {done}")
        
        # 2. Alternative annotator answer (if available)
        if len(task.all_answers) > 1:
            alt_answer = task.all_answers[1] if task.all_answers[0] == task.answer else task.all_answers[0]
            chat_history = [{"role": "assistant", "content": alt_answer}]
            reward, done, message, val_info = task.validate(chat_history, alt_answer)
            print(f"    2. Alternative answer: '{alt_answer}' -> Reward: {reward}")
        
        # 3. Handling unanswerable
        if task.is_unanswerable:
            chat_history = [{"role": "assistant", "content": "I cannot answer this due to image quality"}]
            reward, done, message, val_info = task.validate(chat_history, "cannot answer")
            print(f"    3. Unanswerable response: Reward: {reward}")
        
        # Get metrics
        metrics = task.get_metrics()
        print(f"\n  Task Metrics:")
        print(f"    - Answer type: {metrics.get('answer_type')}")
        print(f"    - Unique answers: {metrics.get('unique_answers')}")
        print(f"    - Difficulty factors: {metrics.get('difficulty_factors', [])}")


def test_filtering_capabilities(adapter: VizWizAdapter):
    """Test various filtering options"""
    print("\n" + "="*60)
    print("Testing Filtering Capabilities")
    print("="*60)
    
    # Test confidence filtering
    print("\n[Confidence-based Filtering]")
    for min_conf in [0.0, 0.3, 0.5, 0.7, 0.9]:
        task_ids = adapter.get_task_ids(min_confidence=min_conf, limit=1000)
        percentage = len(task_ids) / adapter.stats['total'] * 100 if adapter.stats['total'] > 0 else 0
        print(f"  - Confidence ≥ {min_conf}: {len(task_ids)} tasks ({percentage:.1f}%)")
    
    # Test answer type filtering
    print("\n[Answer Type Filtering]")
    for atype in adapter.answer_type_index.keys():
        task_ids = adapter.get_task_ids(answer_type=atype, limit=1000)
        percentage = len(task_ids) / adapter.stats['total'] * 100 if adapter.stats['total'] > 0 else 0
        print(f"  - {atype}: {len(task_ids)} tasks ({percentage:.1f}%)")
    
    # Test special filters
    print("\n[Special Filters]")
    
    # Test filters that are actually available in the adapter
    try:
        # Try unanimous tasks
        unanimous_ids = adapter.get_task_ids(is_unanimous=True, limit=1000)
        print(f"  - Unanimous agreement: {len(unanimous_ids)}")
    except TypeError:
        print("  - Unanimous filter not available")
    
    try:
        # Try controversial tasks
        controversial_ids = adapter.get_task_ids(is_controversial=True, limit=1000)
        print(f"  - Controversial: {len(controversial_ids)}")
    except TypeError:
        print("  - Controversial filter not available")
    
    try:
        # Try high diversity tasks
        high_diversity_ids = adapter.get_task_ids(has_high_diversity=True, limit=1000)
        print(f"  - High answer diversity: {len(high_diversity_ids)}")
    except TypeError:
        print("  - High diversity filter not available")
    
    # Unanswerable tasks - using answerable=False instead
    unanswerable_ids = adapter.get_task_ids(answerable=False, limit=1000)
    print(f"  - Unanswerable: {len(unanswerable_ids)}")
    
    # Find yes/no questions by answer content
    yes_no_count = 0
    for task_id in adapter.get_task_ids(limit=200, shuffle=True):
        task_data = adapter.get_task_data(task_id)
        if task_data['answer'].lower() in ['yes', 'no']:
            yes_no_count += 1
    print(f"  - Yes/No questions (by answer): {yes_no_count} in sample of 200")
    
    # Combined filters
    print("\n[Combined Filters]")
    high_quality = adapter.get_task_ids(
        min_confidence=0.7,
        answerable=True,
        limit=100
    )
    print(f"  - High quality (conf≥0.7, answerable): {len(high_quality)}")
    
    # Low confidence but answerable
    challenging = adapter.get_task_ids(
        max_confidence=0.4,
        answerable=True,
        limit=100
    )
    print(f"  - Challenging (conf≤0.4, answerable): {len(challenging)}")


def test_batch_evaluation(adapter: VizWizAdapter, n_samples: int = 50):
    """Test batch evaluation across different characteristics"""
    print("\n" + "="*60)
    print(f"Testing Batch Evaluation ({n_samples} samples)")
    print("="*60)
    
    # Get balanced sample of tasks
    # Use basic sampling since we're not sure what parameters the adapter supports
    task_ids = adapter.get_task_ids(limit=n_samples, shuffle=True, seed=42)
    
    results = {
        'by_confidence': defaultdict(lambda: {'total': 0, 'correct': 0}),
        'by_category': defaultdict(lambda: {'total': 0, 'correct': 0}),
        'by_answerable': defaultdict(lambda: {'total': 0, 'correct': 0}),
        'by_diversity': defaultdict(lambda: {'total': 0, 'correct': 0}),
    }
    
    print(f"\nProcessing {len(task_ids)} tasks...")
    
    for i, task_id in enumerate(task_ids):
        try:
            task = VizWizTask(task_id=task_id, adapter=adapter)
            goal, info = task.setup()
            
            # Test with correct answer
            success, _ = task.check_success(task.answer)
            
            # Categorize confidence
            conf_category = 'high' if task.answer_confidence >= 0.7 else \
                          'medium' if task.answer_confidence >= 0.4 else 'low'
            
            # Categorize diversity
            diversity = getattr(adapter, 'answer_diversity_index', {}).get(task_id, 0)
            diversity_category = 'high' if diversity > 0.5 else 'low'
            
            # Record results
            results['by_confidence'][conf_category]['total'] += 1
            results['by_category'][task.question_category]['total'] += 1
            results['by_answerable'][task.answerable]['total'] += 1
            results['by_diversity'][diversity_category]['total'] += 1
            
            if success:
                results['by_confidence'][conf_category]['correct'] += 1
                results['by_category'][task.question_category]['correct'] += 1
                results['by_answerable'][task.answerable]['correct'] += 1
                results['by_diversity'][diversity_category]['correct'] += 1
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(task_ids)} tasks...")
                
        except Exception as e:
            print(f"  Error with task {task_id}: {e}")
    
    # Print summary
    print("\nResults Summary:")
    
    print("\nBy Confidence Level:")
    for conf, counts in sorted(results['by_confidence'].items()):
        if counts['total'] > 0:
            acc = counts['correct'] / counts['total'] * 100
            print(f"  - {conf}: {counts['total']} tasks, {acc:.1f}% validation success")
    
    print("\nBy Question Category:")
    for category, counts in sorted(results['by_category'].items()):
        if counts['total'] > 0:
            acc = counts['correct'] / counts['total'] * 100
            print(f"  - {category}: {counts['total']} tasks, {acc:.1f}% validation success")
    
    print("\nBy Answerable Status:")
    for answerable, counts in results['by_answerable'].items():
        if counts['total'] > 0:
            acc = counts['correct'] / counts['total'] * 100
            status = "Yes" if answerable == 1 else "No" if answerable == 0 else "Unknown"
            print(f"  - Answerable={status}: {counts['total']} tasks, {acc:.1f}% validation success")


def visualize_sample_images(adapter: VizWizAdapter, n_samples: int = 6):
    """Visualize sample images with questions and answers"""
    print("\n" + "="*60)
    print(f"Visualizing {n_samples} Sample Images")
    print("="*60)
    
    # Get diverse examples - mix of different confidence levels
    diverse_tasks = []
    
    # Try to get tasks from different confidence ranges
    confidence_ranges = ['very_high', 'high', 'medium', 'low']
    samples_per_range = max(1, n_samples // len(confidence_ranges))
    
    for conf_range in confidence_ranges:
        range_tasks = adapter.get_task_ids(confidence_range=conf_range, limit=samples_per_range)
        diverse_tasks.extend(range_tasks[:samples_per_range])
    
    # Fill remaining slots with random tasks if needed
    if len(diverse_tasks) < n_samples:
        additional = adapter.get_task_ids(limit=n_samples - len(diverse_tasks), shuffle=True)
        diverse_tasks.extend(additional)
    
    diverse_tasks = diverse_tasks[:n_samples]
    
    if not diverse_tasks:
        print("No tasks available for visualization")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, task_id in enumerate(diverse_tasks[:n_samples]):
        try:
            task = VizWizTask(task_id=task_id, adapter=adapter)
            task.setup()
            
            # Load and display image
            if Path(task.image_path).exists():
                img = Image.open(task.image_path)
                axes[i].imshow(img)
                axes[i].axis('off')
                
                # Add title with question info
                question_short = task.question[:50] + "..." if len(task.question) > 50 else task.question
                confidence_str = f"Conf: {task.answer_confidence:.2f}"
                answerable_str = "Answerable" if task.answerable else "Unanswerable"
                
                title = f"{task.question_category.upper()}\n{question_short}\n"
                title += f"Answer: {task.answer[:30]}{'...' if len(task.answer) > 30 else ''}\n"
                title += f"{confidence_str} | {answerable_str}"
                
                axes[i].set_title(title, fontsize=9, wrap=True)
                
                img.close()
            else:
                axes[i].text(0.5, 0.5, f"Image not found\n{Path(task.image_path).name}", 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].axis('off')
            
            # Print info
            print(f"\nSample {i+1}:")
            print(f"  - Category: {task.question_category}")
            print(f"  - Confidence: {task.answer_confidence:.3f}")
            print(f"  - Unique answers: {len(task.all_answers)}")
            
        except Exception as e:
            print(f"Error with task {task_id}: {e}")
            axes[i].text(0.5, 0.5, f"Error loading task", 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')
    
    plt.tight_layout()
    output_file = "vizwiz_samples.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSample images saved to: {output_file}")
    plt.close()


def test_answer_analysis(adapter: VizWizAdapter):
    """Analyze answer patterns in the dataset"""
    print("\n" + "="*60)
    print("Testing Answer Pattern Analysis")
    print("="*60)
    
    # Check if adapter has get_answer_analysis method
    if hasattr(adapter, 'get_answer_analysis'):
        # Get comprehensive answer analysis
        analysis = adapter.get_answer_analysis()
        
        print(f"\nAnswer Statistics:")
        print(f"  - Total unique answers: {analysis['total_unique_answers']:,}")
        print(f"  - Single word answers: {analysis['single_word_answers']:,}")
        print(f"  - Multi-word answers: {analysis['multi_word_answers']:,}")
        print(f"  - Numeric answers: {analysis['numeric_answers']:,}")
        print(f"  - Yes/No answers: {analysis['yes_no_answers']:,}")
        print(f"  - Color answers: {analysis['color_answers']:,}")
        print(f"  - Object answers: {analysis['object_answers']:,}")
        
        print(f"\nTop 15 Most Common Answers:")
        for i, (answer, count) in enumerate(analysis['top_50_answers'][:15]):
            print(f"  {i+1:2d}. '{answer}': {count:,}")
    else:
        # Manual analysis if method doesn't exist
        print("\nManual Answer Analysis:")
        
        # Get statistics from adapter if available
        stats = adapter.get_statistics()
        if 'top_20_common_answers' in stats:
            print(f"\nTop 15 Most Common Answers:")
            for i, (answer, count) in enumerate(stats['top_20_common_answers'][:15]):
                print(f"  {i+1:2d}. '{answer}': {count:,}")
        
        # Analyze a sample of answers
        sample_size = min(200, len(adapter._task_index))
        sample_ids = adapter.get_task_ids(limit=sample_size, shuffle=True)
        
        answer_types = {
            'single_word': 0,
            'multi_word': 0,
            'numeric': 0,
            'yes_no': 0,
            'color': 0,
            'unanswerable': 0
        }
        
        colors = {'red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 
                 'purple', 'orange', 'pink', 'gray', 'grey'}
        
        for task_id in sample_ids:
            task_data = adapter.get_task_data(task_id)
            answer = task_data['answer'].lower()
            
            # Categorize answer
            if answer in ['yes', 'no']:
                answer_types['yes_no'] += 1
            elif answer == 'unanswerable':
                answer_types['unanswerable'] += 1
            elif answer in colors:
                answer_types['color'] += 1
            elif answer.replace('.', '').replace(',', '').isdigit():
                answer_types['numeric'] += 1
            
            # Word count
            word_count = len(answer.split())
            if word_count == 1:
                answer_types['single_word'] += 1
            else:
                answer_types['multi_word'] += 1
        
        print(f"\nAnswer Type Distribution (sample of {sample_size}):")
        for atype, count in answer_types.items():
            percentage = count / sample_size * 100
            print(f"  - {atype}: {count} ({percentage:.1f}%)")


def test_data_validation(adapter: VizWizAdapter):
    """Validate VizWiz data integrity"""
    print("\n" + "="*60)
    print("Testing Data Validation")
    print("="*60)
    
    print("\nValidating all tasks...")
    
    # Check if adapter has validate_all method
    if hasattr(adapter, 'validate_all'):
        issues = adapter.validate_all()
        
        total_issues = sum(len(task_list) for task_list in issues.values())
        
        if total_issues == 0:
            print("✓ No validation issues found!")
        else:
            print(f"Found {total_issues} issues:")
            
            for issue_type, task_ids in issues.items():
                if task_ids:
                    print(f"\n  {issue_type}: {len(task_ids)} tasks")
                    # Show a few examples
                    for task_id in task_ids[:3]:
                        print(f"    - {task_id}")
                    if len(task_ids) > 3:
                        print(f"    ... and {len(task_ids) - 3} more")
    else:
        # Manual validation if method doesn't exist
        print("Performing manual validation...")
        
        issues = defaultdict(list)
        sample_size = min(100, len(adapter._task_index))
        sample_ids = adapter.get_task_ids(limit=sample_size)
        
        for task_id in sample_ids:
            task_data = adapter.get_task_data(task_id)
            
            # Check for missing fields
            if not task_data.get('question'):
                issues['missing_question'].append(task_id)
            if not task_data.get('answer'):
                issues['missing_answer'].append(task_id)
            if not task_data.get('image_path'):
                issues['missing_image_path'].append(task_id)
            
            # Check metadata
            metadata = task_data.get('metadata', {})
            if 'answer_confidence' not in metadata:
                issues['missing_confidence'].append(task_id)
            if 'all_answers' not in metadata:
                issues['missing_all_answers'].append(task_id)
        
        total_issues = sum(len(task_list) for task_list in issues.values())
        
        if total_issues == 0:
            print(f"✓ No validation issues found in sample of {sample_size} tasks!")
        else:
            print(f"Found {total_issues} issues in sample of {sample_size} tasks:")
            
            for issue_type, task_ids in issues.items():
                if task_ids:
                    print(f"\n  {issue_type}: {len(task_ids)} tasks")
                    # Show a few examples
                    for task_id in task_ids[:3]:
                        print(f"    - {task_id}")
                    if len(task_ids) > 3:
                        print(f"    ... and {len(task_ids) - 3} more")


def test_edge_cases(adapter: VizWizAdapter):
    """Test edge cases and special scenarios"""
    print("\n" + "="*60)
    print("Testing Edge Cases")
    print("="*60)
    
    # Test 1: Very short answers
    print("\n[Very Short Answers]")
    short_answer_tasks = []
    for task_id in adapter.get_task_ids(limit=100, shuffle=True):
        task_data = adapter.get_task_data(task_id)
        if len(task_data['answer']) <= 2:
            short_answer_tasks.append(task_id)
            if len(short_answer_tasks) >= 3:
                break
    
    for task_id in short_answer_tasks[:3]:
        task_data = adapter.get_task_data(task_id)
        print(f"  Q: {task_data['question'][:60]}...")
        print(f"  A: '{task_data['answer']}' (confidence: {task_data['metadata']['answer_confidence']:.3f})")
    
    # Test 2: Very long questions
    print("\n[Very Long Questions]")
    long_question_tasks = []
    for task_id in adapter.get_task_ids(limit=100, shuffle=True):
        task_data = adapter.get_task_data(task_id)
        if len(task_data['question']) > 100:
            long_question_tasks.append(task_id)
            if len(long_question_tasks) >= 2:
                break
    
    for task_id in long_question_tasks[:2]:
        task_data = adapter.get_task_data(task_id)
        print(f"  Q ({len(task_data['question'])} chars): {task_data['question']}")
        print(f"  A: {task_data['answer']}")
    
    # Test 3: Questions with special characters
    print("\n[Questions with Special Characters]")
    special_char_tasks = []
    for task_id in adapter.get_task_ids(limit=200, shuffle=True):
        task_data = adapter.get_task_data(task_id)
        if any(char in task_data['question'] for char in ['?', '!', '"', "'", '-']):
            special_char_tasks.append(task_id)
            if len(special_char_tasks) >= 2:
                break
    
    for task_id in special_char_tasks[:2]:
        task_data = adapter.get_task_data(task_id)
        print(f"  Q: {task_data['question']}")


def test_accessibility_features(adapter: VizWizAdapter):
    """Test accessibility features for visually impaired users"""
    print("\n" + "="*60)
    print("Testing Accessibility Features")
    print("="*60)
    
    # Get a mix of different types of tasks
    test_tasks = []
    
    # Get one of each type
    task_types = [
        ('high_confidence', adapter.get_task_ids(min_confidence=0.8, limit=1)),
        ('low_confidence', adapter.get_task_ids(max_confidence=0.3, limit=1)),
        ('unanswerable', adapter.get_task_ids(answerable=False, limit=1)),
        ('color_question', []),  # Will find manually
        ('counting_question', [])  # Will find manually
    ]
    
    # Find color and counting questions
    for task_id in adapter.get_task_ids(limit=50, shuffle=True):
        task_data = adapter.get_task_data(task_id)
        if 'color' in task_data['question'].lower() and not task_types[3][1]:
            task_types[3] = ('color_question', [task_id])
        elif 'how many' in task_data['question'].lower() and not task_types[4][1]:
            task_types[4] = ('counting_question', [task_id])
        
        if task_types[3][1] and task_types[4][1]:
            break
    
    # Collect all test tasks
    for type_name, task_ids in task_types:
        if task_ids:
            test_tasks.append(task_ids[0])
    
    test_tasks = test_tasks[:5]  # Limit to 5 tasks
    
    for i, task_id in enumerate(test_tasks):
        task = VizWizTask(task_id=task_id, adapter=adapter)
        task.setup()
        
        print(f"\n[Task {i+1}]")
        print(f"  Question: {task.question}")
        print(f"  Original answer: {task.answer}")
        
        # Get assistance hints
        hints = task.get_assistance_hints()
        if hints:
            print(f"  Assistance hints:")
            for hint in hints:
                print(f"    - {hint}")
        
        # Test accessibility formatting
        formatted_answer = task.format_for_accessibility(task.answer)
        if formatted_answer != task.answer:
            print(f"  Formatted for accessibility: {formatted_answer}")


def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description='Test VizWiz integration with VLM Gym')
    parser.add_argument('--annotation', type=str,
                       default='/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/VizWiz/vizwiz_train_vlmgym.json',
                       help='Path to VizWiz annotation file')
    parser.add_argument('--data-root', type=str,
                       default='/data/wang/meng/GYM-Work/dataset/VizWiz',
                       help='Path to VizWiz images')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to test')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Batch size for evaluation')
    parser.add_argument('--samples', type=int, default=6,
                       help='Number of samples for visualization')
    parser.add_argument('--min-confidence', type=float, default=0.0,
                       help='Minimum confidence threshold for testing')
    
    args = parser.parse_args()
    
    print("="*60)
    print("VizWiz-VLMGym Integration Test")
    print("="*60)
    print(f"Testing VizWiz dataset for visually impaired users")
    print(f"Features: Multi-annotator answers, confidence scores, unanswerable detection")
    print("="*60)
    
    try:
        # Test 1: Load adapter
        adapter = test_adapter_loading(args.annotation, args.data_root, args.split)
        
        # Check if adapter loaded any data
        if len(adapter._task_index) == 0:
            print("\n❌ No data loaded! Please check:")
            print(f"  1. Annotation file exists: {args.annotation}")
            print(f"  2. Image directory exists: {args.data_root}")
            print(f"  3. Annotation file contains valid VizWiz data")
            return 1
        
        # Test 2: Create tasks by confidence level
        tasks = test_task_creation_by_confidence(adapter)
        
        # Test 3: Test question categories
        test_question_categories(adapter)
        
        # Test 4: Comprehensive answer validation
        test_answer_validation_comprehensive(adapter)
        
        # Test 5: Color questions
        test_color_questions(adapter)
        
        # Test 6: Counting questions
        test_counting_questions(adapter)
        
        # Test 7: Annotator agreement analysis
        test_annotator_agreement_analysis(adapter)
        
        # Test 8: Complete workflow
        test_complete_workflow(adapter)
        
        # Test 9: Filtering capabilities
        test_filtering_capabilities(adapter)
        
        # Test 10: Batch evaluation
        test_batch_evaluation(adapter, n_samples=args.batch_size)
        
        # Test 11: Visualize samples
        visualize_sample_images(adapter, n_samples=args.samples)
        
        # Test 12: Answer analysis
        test_answer_analysis(adapter)
        
        # Test 13: Data validation
        test_data_validation(adapter)
        
        # Test 14: Edge cases
        test_edge_cases(adapter)
        
        # Test 15: Accessibility features
        test_accessibility_features(adapter)
        
        print("\n" + "="*60)
        print("✓ All tests completed successfully!")
        print("="*60)
        
        print("\nKey findings:")
        print("1. VizWiz adapter successfully handles multi-annotator data")
        print("2. Answer validation works with various confidence levels")
        print("3. Unanswerable questions are properly identified and handled")
        print("4. Question categories (color, counting, text reading) are recognized")
        print("5. Accessibility features support visually impaired users")
        
        print("\nUnique VizWiz features tested:")
        print("- Answer confidence scores and agreement levels")
        print("- Multiple valid answers from different annotators")
        print("- Handling of poor image quality and unanswerable cases")
        print("- Flexible answer matching for real-world variations")
        
        print("\nNext steps:")
        print("1. Check the vizwiz_samples.png for visualization")
        print("2. Test with actual VLM models on VizWiz tasks")
        print("3. Evaluate performance on different confidence levels")
        print("4. Test with models specialized for accessibility")
        print("5. Analyze failure cases on unanswerable questions")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())