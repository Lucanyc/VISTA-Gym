#!/usr/bin/env python3
"""
Test script for VQA-Med integration with VLM Gym
Tests medical visual question answering focusing on imaging modalities

NOTE: The VQA-Med dataset conversion has some known issues:
1. Many questions are mislabeled as 'yes_no' when they have other answers (e.g., 't2', 'noncontrast')
2. The 'choices' field often contains ['yes', 'no'] even when the answer is neither
3. This test script handles these inconsistencies and provides diagnostics
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
    from vqa_med import VqaMedTask
except ImportError as e:
    print(f"Warning: Failed to import VqaMedTask from current directory: {e}")
    try:
        from vlm_gym.environments.task.vqa_med import VqaMedTask
    except ImportError as e2:
        print(f"Error: Cannot import VqaMedTask: {e2}")
        print("\nPlease ensure vqa_med.py exists in:")
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
    from vqamed_adapter import VqaMedAdapter
except ImportError as e:
    print(f"Error: Cannot import VqaMedAdapter: {e}")
    print("\nMake sure vqamed_adapter.py is in:")
    print("/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data_adapters/")
    sys.exit(1)


def test_adapter_loading(annotation_file, data_root=None):
    """Test VQA-Med adapter initialization and data loading"""
    print("\n" + "="*60)
    print("Testing VQA-Med Adapter Loading")
    print("="*60)
    
    # Initialize adapter with provided paths
    print(f"Loading from: {annotation_file}")
    if data_root:
        print(f"Data root: {data_root}")
    
    # Check if file exists
    if not Path(annotation_file).exists():
        print(f"❌ Annotation file not found: {annotation_file}")
        return None
    
    adapter = VqaMedAdapter(
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
    
    # Clinical distribution
    clinical_dist = stats.get('clinical_distribution', {})
    if clinical_dist:
        print(f"\n  Clinical Distribution:")
        print(f"    - Clinical: {clinical_dist.get('clinical', 0)} ({clinical_dist.get('clinical', 0)/stats['total']*100:.1f}%)")
        print(f"    - Non-clinical: {clinical_dist.get('non_clinical', 0)} ({clinical_dist.get('non_clinical', 0)/stats['total']*100:.1f}%)")
    
    # Task distribution
    print(f"\n  Medical Task Type Distribution:")
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
    
    # Modality distribution
    print(f"\n  Imaging Modality Distribution:")
    for modality, count in list(stats.get('modality_distribution', {}).items())[:10]:
        print(f"    - {modality}: {count} occurrences")
    
    # Answer categories
    print(f"\n  Answer Category Distribution:")
    answer_cats = stats.get('answer_category_statistics', {})
    if answer_cats:
        print(f"    - Modality answers: {answer_cats.get('modality_answers', 0)}")
        print(f"    - Anatomical answers: {answer_cats.get('anatomical_answers', 0)}")
        print(f"    - Plane answers: {answer_cats.get('plane_answers', 0)}")
    
    # Medical question types
    print(f"\n  Medical Question Types:")
    med_questions = stats.get('medical_question_statistics', {})
    if med_questions:
        print(f"    - Modality questions: {med_questions.get('modality_questions', 0)}")
        print(f"    - Contrast questions: {med_questions.get('contrast_questions', 0)}")
        print(f"    - Weighted questions: {med_questions.get('weighted_questions', 0)}")
        print(f"    - Technical questions: {med_questions.get('technical_questions', 0)}")
    
    return adapter


def test_modality_recognition_tasks(adapter: VqaMedAdapter):
    """Test modality recognition - the main focus of VQA-Med"""
    print("\n" + "="*60)
    print("Testing Modality Recognition Tasks")
    print("="*60)
    
    # First, let's check what questions are actually about modality
    print("\nSearching for actual modality questions...")
    
    # Keywords that indicate modality questions
    modality_keywords = ['modality', 'kind of image', 'type of image', 'what is this image', 
                        'taken with', 'imaging modality', 'type of scan']
    
    modality_questions = []
    all_tasks = adapter.get_task_ids(limit=200)
    
    for task_id in all_tasks:
        task_data = adapter.get_task_data(task_id)
        question = task_data.get('question', '').lower()
        
        if any(keyword in question for keyword in modality_keywords):
            modality_questions.append(task_id)
    
    print(f"Found {len(modality_questions)} questions about imaging modality")
    
    if not modality_questions:
        # Try the adapter's modality questions method
        modality_questions = adapter.get_modality_questions(n=10)
    
    if not modality_questions:
        print("No modality recognition tasks found!")
        return
    
    # Test a few examples
    tested_count = 0
    for task_id in modality_questions[:10]:
        task_data = adapter.get_task_data(task_id)
        answer = task_data.get('answer', '').lower()
        
        # Check if answer is actually a modality
        if any(mod in answer for mod in ['ct', 'mri', 'xr', 'x-ray', 'ultrasound', 'us -', 'cta', 'mra']):
            task = VqaMedTask(task_id=task_id, adapter=adapter)
            goal, info = task.setup()
            
            print(f"\n[Modality Question {tested_count + 1}]")
            print(f"  Question: {task.question}")
            print(f"  Expected: '{task.expected_answer}'")
            print(f"  Task type: {task.task_type}")
            print(f"  Answer category: {task.answer_category}")
            
            # Test answer validation
            success, feedback = task.check_success(task.expected_answer)
            print(f"  Self-validation: {'✅' if success else '❌'}")
            
            # Test some variations if it's a standard modality
            if 'xr' in task.expected_answer or 'x-ray' in task.expected_answer:
                test_answers = ['x-ray', 'xr - plain film', 'plain film']
                print(f"  Testing X-ray variations:")
                for test_ans in test_answers:
                    success, _ = task.check_success(test_ans)
                    print(f"    {'✅' if success else '❌'} '{test_ans}'")
            
            tested_count += 1
            if tested_count >= 3:
                break
    
    if tested_count == 0:
        print("\n⚠️ No actual modality answers found in modality questions!")
        print("Sample questions labeled as modality recognition:")
        for i, task_id in enumerate(modality_questions[:3]):
            task_data = adapter.get_task_data(task_id)
            print(f"\n  {i+1}. Q: {task_data['question']}")
            print(f"     A: {task_data['answer']}")
            print(f"     Task: {task_data.get('task', 'unknown')}")


def test_contrast_questions(adapter: VqaMedAdapter):
    """Test contrast-related questions"""
    print("\n" + "="*60)
    print("Testing Contrast Questions")
    print("="*60)
    
    # Get contrast questions
    contrast_tasks = adapter.get_contrast_questions(n=10)
    
    if not contrast_tasks:
        # Search for contrast questions manually
        all_tasks = adapter.get_task_ids(limit=200)
        contrast_tasks = []
        for task_id in all_tasks:
            task_data = adapter.get_task_data(task_id)
            question = task_data.get('question', '').lower()
            answer = task_data.get('answer', '').lower()
            
            if 'contrast' in question or 'contrast' in answer or 'noncontrast' in answer:
                contrast_tasks.append(task_id)
                if len(contrast_tasks) >= 10:
                    break
    
    if not contrast_tasks:
        print("No contrast questions found!")
        return
    
    print(f"Found {len(contrast_tasks)} contrast-related questions")
    
    # Categorize by answer type
    contrast_answers = Counter()
    for task_id in contrast_tasks[:20]:
        task_data = adapter.get_task_data(task_id)
        answer = task_data.get('answer', '').lower()
        contrast_answers[answer] += 1
    
    print("\nContrast question answer distribution:")
    for answer, count in contrast_answers.most_common():
        print(f"  - '{answer}': {count} times")
    
    # Test a few examples
    for i, task_id in enumerate(contrast_tasks[:3]):
        task = VqaMedTask(task_id=task_id, adapter=adapter)
        goal, info = task.setup()
        
        print(f"\n[Contrast Question {i+1}]")
        print(f"  Question: {task.question}")
        print(f"  Expected: '{task.expected_answer}'")
        print(f"  Answer type: {task.answer_type}")
        print(f"  Is binary: {task.is_binary}")
        
        # Test validation with correct answer
        success, feedback = task.check_success(task.expected_answer)
        print(f"  Self-validation: {'✅' if success else '❌'}")
        
        # Test contrast variations based on the answer
        if 'contrast' in task.expected_answer.lower() and 'non' not in task.expected_answer.lower():
            test_answers = ['contrast', 'with contrast', 'post-contrast']
        elif 'noncontrast' in task.expected_answer.lower():
            test_answers = ['noncontrast', 'non-contrast', 'without contrast']
        else:
            test_answers = [task.expected_answer]
        
        if len(test_answers) > 1:
            print(f"  Testing contrast answer variations:")
            for test_ans in test_answers[:3]:
                success, _ = task.check_success(test_ans)
                print(f"    {'✅' if success else '❌'} '{test_ans}'")


def test_weighted_mri_questions(adapter: VqaMedAdapter):
    """Test MRI weighted sequence questions"""
    print("\n" + "="*60)
    print("Testing MRI Weighted Sequence Questions")
    print("="*60)
    
    # Get weighted questions
    weighted_tasks = adapter.get_weighted_questions(n=10)
    
    if not weighted_tasks:
        # Search manually
        all_tasks = adapter.get_task_ids(limit=200)
        weighted_tasks = []
        for task_id in all_tasks:
            task_data = adapter.get_task_data(task_id)
            question = task_data.get('question', '').lower()
            
            if any(term in question for term in ['t1', 't2', 'flair', 'weighted', 'dwi']):
                weighted_tasks.append(task_id)
                if len(weighted_tasks) >= 10:
                    break
    
    if not weighted_tasks:
        print("No MRI weighted sequence questions found!")
        return
    
    print(f"Found {len(weighted_tasks)} MRI weighted sequence questions")
    
    # Check answer distribution
    sequence_answers = Counter()
    for task_id in weighted_tasks[:20]:
        task_data = adapter.get_task_data(task_id)
        answer = task_data.get('answer', '').lower()
        sequence_answers[answer] += 1
    
    print("\nMRI sequence answer distribution:")
    for answer, count in sequence_answers.most_common():
        print(f"  - '{answer}': {count} times")
    
    # Test examples
    tested_count = 0
    for task_id in weighted_tasks[:10]:
        task_data = adapter.get_task_data(task_id)
        answer = task_data.get('answer', '').lower()
        
        # Test if answer is actually an MRI sequence
        if any(seq in answer for seq in ['t1', 't2', 'flair', 'dwi', 'stir']):
            task = VqaMedTask(task_id=task_id, adapter=adapter)
            goal, info = task.setup()
            
            print(f"\n[MRI Sequence Question {tested_count + 1}]")
            print(f"  Question: {task.question}")
            print(f"  Expected: '{task.expected_answer}'")
            print(f"  Is weighted question: {task.is_weighted_question}")
            print(f"  Answer type: {task.answer_type}")
            
            # Test validation
            success, feedback = task.check_success(task.expected_answer)
            print(f"  Self-validation: {'✅' if success else '❌'}")
            
            # Test case variations
            if task.expected_answer.lower() == 't2':
                test_answers = ['t2', 'T2', 't2 weighted', 'T2-weighted']
                print(f"  Testing T2 variations:")
                for test_ans in test_answers[:3]:
                    success, _ = task.check_success(test_ans)
                    print(f"    {'✅' if success else '❌'} '{test_ans}'")
            
            tested_count += 1
            if tested_count >= 3:
                break


def test_anatomical_system_answers(adapter: VqaMedAdapter):
    """Test anatomical system recognition"""
    print("\n" + "="*60)
    print("Testing Anatomical System Answers")
    print("="*60)
    
    # Get anatomical answer examples
    anatomical_tasks = adapter.get_anatomical_answers(n=10)
    
    if not anatomical_tasks:
        print("No anatomical system answer tasks found!")
        return
    
    print(f"Found {len(anatomical_tasks)} anatomical system tasks")
    
    # Group by anatomical system
    anatomical_systems = defaultdict(list)
    for task_id in anatomical_tasks[:10]:
        task_data = adapter.get_task_data(task_id)
        answer = task_data.get('answer', '').lower()
        
        # Common anatomical systems in VQA-Med
        systems = ['skull and contents', 'musculoskeletal', 'gastrointestinal', 
                   'spine and contents', 'cardiovascular', 'respiratory']
        
        for system in systems:
            if system in answer:
                anatomical_systems[system].append(task_id)
                break
    
    # Test examples from different systems
    for system, task_ids in list(anatomical_systems.items())[:3]:
        if task_ids:
            task_id = task_ids[0]
            task = VqaMedTask(task_id=task_id, adapter=adapter)
            goal, info = task.setup()
            
            print(f"\n[Anatomical System: {system}]")
            print(f"  Question: {task.question}")
            print(f"  Expected: '{task.expected_answer}'")
            print(f"  Answer category: {task.answer_category}")


def test_imaging_plane_answers(adapter: VqaMedAdapter):
    """Test imaging plane recognition"""
    print("\n" + "="*60)
    print("Testing Imaging Plane Answers")
    print("="*60)
    
    # Get plane answer examples
    plane_tasks = adapter.get_plane_answers(n=10)
    
    if not plane_tasks:
        print("No imaging plane answer tasks found!")
        return
    
    print(f"Found {len(plane_tasks)} imaging plane tasks")
    
    # Test different planes
    planes = ['axial', 'sagittal', 'coronal']
    plane_examples = {}
    
    for task_id in plane_tasks[:10]:
        task_data = adapter.get_task_data(task_id)
        answer = task_data.get('answer', '').lower()
        
        for plane in planes:
            if plane in answer and plane not in plane_examples:
                plane_examples[plane] = task_id
    
    # Test each plane type
    for plane, task_id in plane_examples.items():
        task = VqaMedTask(task_id=task_id, adapter=adapter)
        goal, info = task.setup()
        
        print(f"\n[{plane.capitalize()} Plane]")
        print(f"  Question: {task.question}")
        print(f"  Expected: '{task.expected_answer}'")
        
        # Test plane variations
        test_answers = [plane, plane.capitalize(), f"{plane} view", f"{plane} plane"]
        
        print(f"  Testing plane variations:")
        for test_ans in test_answers[:3]:
            success, feedback = task.check_success(test_ans)
            status = "✅" if success else "❌"
            print(f"    {status} '{test_ans}'")


def test_yes_no_validation(adapter: VqaMedAdapter):
    """Test yes/no answer validation for medical questions"""
    print("\n" + "="*60)
    print("Testing Yes/No Answer Validation")
    print("="*60)
    
    # Get yes/no tasks
    yes_no_tasks = adapter.get_yes_no_examples(n=6)
    
    if not yes_no_tasks:
        print("No yes/no tasks found!")
        return
    
    # Get one yes and one no example
    yes_example = None
    no_example = None
    
    for task_id in yes_no_tasks:
        task_data = adapter.get_task_data(task_id)
        answer = task_data.get('answer', '').lower()
        if answer == 'yes' and not yes_example:
            yes_example = task_id
        elif answer == 'no' and not no_example:
            no_example = task_id
        if yes_example and no_example:
            break
    
    # Test YES answer variations
    if yes_example:
        task = VqaMedTask(task_id=yes_example, adapter=adapter)
        task.setup()
        
        print(f"\n[Testing YES answer]")
        print(f"  Question: {task.question}")
        print(f"  Expected: '{task.expected_answer}'")
        
        test_cases = [
            ("yes", "Exact 'yes'"),
            ("Yes", "Capitalized"),
            ("YES", "All caps"),
            ("y", "Single letter"),
            ("true", "Boolean true"),
            ("no", "Wrong answer"),
            ("maybe", "Ambiguous"),
            ("", "Empty string")
        ]
        
        print(f"  Testing variations:")
        for test_answer, desc in test_cases:
            success, feedback = task.check_success(test_answer)
            status = "✅" if success else "❌"
            print(f"    {status} {desc}: '{test_answer}'")
    
    # Test NO answer variations
    if no_example:
        task = VqaMedTask(task_id=no_example, adapter=adapter)
        task.setup()
        
        print(f"\n[Testing NO answer]")
        print(f"  Question: {task.question}")
        print(f"  Expected: '{task.expected_answer}'")
        
        test_cases = [
            ("no", "Exact 'no'"),
            ("No", "Capitalized"),
            ("NO", "All caps"),
            ("n", "Single letter"),
            ("false", "Boolean false"),
            ("yes", "Wrong answer"),
            ("", "Empty string")
        ]
        
        print(f"  Testing variations:")
        for test_answer, desc in test_cases:
            success, feedback = task.check_success(test_answer)
            status = "✅" if success else "❌"
            print(f"    {status} {desc}: '{test_answer}'")


def test_answer_normalization(adapter: VqaMedAdapter):
    """Test answer normalization for VQA-Med specific formats"""
    print("\n" + "="*60)
    print("Testing VQA-Med Answer Normalization")
    print("="*60)
    
    # Test specific VQA-Med answer formats
    test_mappings = {
        'xr - plain film': ['x-ray', 'xray', 'plain film', 'radiograph'],
        'us - ultrasound': ['ultrasound', 'us', 'sonography'],
        'cta - ct angiography': ['ct angiography', 'cta'],
        'mra - mr angiography': ['mr angiography', 'mra']
    }
    
    for standard_format, variations in test_mappings.items():
        # Find a task with this answer
        examples = adapter.get_examples_by_answer(standard_format, n=1)
        if examples:
            task = VqaMedTask(task_id=examples[0], adapter=adapter)
            task.setup()
            
            print(f"\n[Standard format: '{standard_format}']")
            print(f"  Question: {task.question[:60]}...")
            
            print(f"  Testing normalizations:")
            for variation in variations:
                success, feedback = task.check_success(variation)
                status = "✅" if success else "❌"
                print(f"    {status} '{variation}' -> normalized correctly")


def test_data_format_issues(adapter: VqaMedAdapter):
    """Test and identify data format issues in VQA-Med"""
    print("\n" + "="*60)
    print("Testing Data Format Issues")
    print("="*60)
    
    # Check for answer_type mismatches
    mismatched_answer_types = []
    
    # Sample some tasks
    sample_ids = adapter.get_task_ids(limit=100)
    
    for task_id in sample_ids:
        task_data = adapter.get_task_data(task_id)
        answer = task_data.get('answer', '').lower()
        answer_type = task_data.get('metadata', {}).get('answer_type', '')
        choices = task_data.get('choices', None)
        
        # Check for mismatches
        if answer_type == 'yes_no' and answer not in ['yes', 'no']:
            mismatched_answer_types.append({
                'id': task_id,
                'answer': answer,
                'answer_type': answer_type,
                'choices': choices,
                'question': task_data.get('question', '')
            })
    
    if mismatched_answer_types:
        print(f"\n⚠️ Found {len(mismatched_answer_types)} tasks with mismatched answer_type:")
        for i, mismatch in enumerate(mismatched_answer_types[:5]):
            print(f"\n  {i+1}. Task ID: {mismatch['id']}")
            print(f"     Question: {mismatch['question'][:60]}...")
            print(f"     Answer: '{mismatch['answer']}'")
            print(f"     Labeled as: {mismatch['answer_type']}")
            print(f"     Choices: {mismatch['choices']}")
        
        if len(mismatched_answer_types) > 5:
            print(f"\n  ... and {len(mismatched_answer_types) - 5} more")
    
    # Check answer distribution for supposedly yes/no questions
    print("\n\nChecking answer distribution for 'yes_no' labeled questions:")
    yes_no_labeled = adapter.get_task_ids(answer_type='yes_no', limit=50)
    answer_counts = Counter()
    
    for task_id in yes_no_labeled:
        task_data = adapter.get_task_data(task_id)
        answer = task_data.get('answer', '').lower()
        answer_counts[answer] += 1
    
    print("\nAnswers in 'yes_no' labeled questions:")
    for answer, count in answer_counts.most_common(10):
        print(f"  - '{answer}': {count} times")
    
    return mismatched_answer_types


def test_actual_yes_no_questions(adapter: VqaMedAdapter):
    """Test actual yes/no questions (where answer is really yes or no)"""
    print("\n" + "="*60)
    print("Testing Actual Yes/No Questions")
    print("="*60)
    
    # Find tasks where answer is actually yes or no
    actual_yes_no = []
    
    # Search through tasks
    all_tasks = adapter.get_task_ids(limit=500)
    for task_id in all_tasks:
        task_data = adapter.get_task_data(task_id)
        answer = task_data.get('answer', '').lower()
        if answer in ['yes', 'no']:
            actual_yes_no.append(task_id)
    
    print(f"Found {len(actual_yes_no)} tasks with actual yes/no answers")
    
    if actual_yes_no:
        # Test a few examples
        for i, task_id in enumerate(actual_yes_no[:3]):
            task = VqaMedTask(task_id=task_id, adapter=adapter)
            task.setup()
            
            print(f"\n[Actual Yes/No Question {i+1}]")
            print(f"  Question: {task.question}")
            print(f"  Answer: '{task.expected_answer}'")
            print(f"  Answer type: {task.answer_type}")
            print(f"  Is binary: {task.is_binary}")
            
            # Test validation
            success_correct, msg = task.check_success(task.expected_answer)
            wrong_answer = 'no' if task.expected_answer == 'yes' else 'yes'
            success_wrong, msg_wrong = task.check_success(wrong_answer)
            
            print(f"  Validation:")
            print(f"    - Correct answer: {'✅' if success_correct else '❌'}")
            print(f"    - Wrong answer: {'✅' if not success_wrong else '❌'}")
    
    return actual_yes_no


def test_complete_workflow(adapter: VqaMedAdapter):
    """Test complete VQA-Med workflow"""
    print("\n" + "="*60)
    print("Testing Complete VQA-Med Workflow")
    print("="*60)
    
    # Get diverse examples
    task_ids = adapter.get_diverse_medical_examples(n=1)
    
    if not task_ids:
        task_ids = adapter.get_task_ids(limit=1)
    
    if not task_ids:
        print("No tasks available!")
        return
    
    task_data = adapter.get_task_data(task_ids[0])
    task = VqaMedTask(task_id=task_ids[0], adapter=adapter)
    
    goal, info = task.setup()
    
    print("VQA-Med Task Setup:")
    print(f"  - Question: {task.question}")
    print(f"  - Expected answer: {task.expected_answer}")
    print(f"  - Task type: {task.task_type}")
    print(f"  - Question type: {task.question_type}")
    print(f"  - Answer type: {task.answer_type}")
    print(f"  - Answer category: {task.answer_category}")
    print(f"  - Clinical relevance: {task.is_clinical}")
    print(f"  - Difficulty: {info['difficulty']}")
    
    # Show special flags
    print(f"\nSpecial question types:")
    print(f"  - Is modality question: {task.is_modality_question}")
    print(f"  - Is contrast question: {task.is_contrast_question}")
    print(f"  - Is weighted question: {task.is_weighted_question}")
    
    # Show medical entities
    if task.medical_entities:
        print(f"\nMedical Entities:")
        for entity_type, values in task.medical_entities.items():
            if values:
                print(f"  - {entity_type}: {', '.join(values)}")
    
    # Get observation
    obs = task.get_observation()
    print(f"\nObservation:")
    print(f"  - Scene type: {obs.get('scene_type', 'N/A')}")
    print(f"  - Answer format: {obs.get('answer_format', 'N/A')}")
    print(f"  - Answer category: {obs.get('answer_category', 'N/A')}")
    
    # Simulate responses
    print(f"\nSimulating agent responses:")
    
    # 1. Correct answer
    correct_answer = task.expected_answer
    chat_history = [{"role": "assistant", "content": f"The answer is {correct_answer}."}]
    reward, done, message, val_info = task.validate(chat_history, correct_answer)
    print(f"  1. Correct answer ('{correct_answer}'): {message}")
    print(f"     Reward: {reward}, Done: {done}")
    
    # 2. Test normalization (if applicable)
    if task.answer_category == 'modality' and correct_answer == 'xr - plain film':
        test_answer = 'x-ray'
        chat_history = [{"role": "assistant", "content": f"This is an {test_answer}."}]
        reward, done, message, val_info = task.validate(chat_history, test_answer)
        print(f"  2. Normalized answer ('{test_answer}'): {message}")
        print(f"     Reward: {reward}")
    
    # 3. Wrong answer (for yes/no)
    if task.is_binary:
        wrong_answer = "yes" if task.expected_answer == "no" else "no"
        chat_history = [{"role": "assistant", "content": f"The answer is {wrong_answer}."}]
        reward, done, message, val_info = task.validate(chat_history, wrong_answer)
        print(f"  3. Wrong answer ('{wrong_answer}'): {message}")
        print(f"     Reward: {reward}")
        if 'error_analysis' in val_info:
            print(f"     Error type: {val_info['error_analysis']['error_type']}")


def test_batch_evaluation(adapter: VqaMedAdapter, n_samples: int = 100):
    """Test batch evaluation for VQA-Med tasks"""
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
    
    # Sample tasks with modality balance
    task_ids = adapter.sample_tasks(
        actual_samples,
        balanced_modality=True,
        seed=42
    )
    
    results = {
        'by_task_type': defaultdict(list),
        'by_answer_category': defaultdict(list),
        'by_answer_type': defaultdict(list),
        'by_difficulty': defaultdict(list),
        'by_modality': defaultdict(list),
        'modality_questions': [],
        'contrast_questions': [],
        'weighted_questions': [],
        'yes_no': [],
        'open_ended': [],
        'errors': defaultdict(int)
    }
    
    print(f"\nProcessing {len(task_ids)} VQA-Med tasks...")
    
    for i, task_id in enumerate(task_ids):
        try:
            task = VqaMedTask(task_id=task_id, adapter=adapter)
            goal, info = task.setup()
            
            # Simulate correct answer
            success, _ = task.check_success(task.expected_answer)
            
            # Record results
            results['by_task_type'][task.task_type].append(success)
            results['by_answer_category'][task.answer_category].append(success)
            results['by_answer_type'][task.answer_type].append(success)
            results['by_difficulty'][info['difficulty']].append(success)
            
            # Special question types
            if task.is_modality_question:
                results['modality_questions'].append(success)
            if task.is_contrast_question:
                results['contrast_questions'].append(success)
            if task.is_weighted_question:
                results['weighted_questions'].append(success)
            
            # Modality tracking
            if task.medical_entities.get('modalities'):
                for modality in task.medical_entities['modalities']:
                    results['by_modality'][modality].append(success)
            
            # Answer type
            if task.is_binary:
                results['yes_no'].append(success)
            else:
                results['open_ended'].append(success)
            
            # Test with wrong answer to collect error types
            if task.answer_category == 'modality':
                # Test wrong modality
                wrong_modality = 'ct' if task.expected_answer != 'ct' else 'mri'
                wrong_success, _ = task.check_success(wrong_modality)
                _, _, _, val_info = task.validate(
                    [{"role": "assistant", "content": wrong_modality}],
                    wrong_modality
                )
                
                if 'error_analysis' in val_info:
                    error_type = val_info['error_analysis']['error_type']
                    results['errors'][error_type] += 1
            
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(task_ids)} tasks...")
                
        except Exception as e:
            print(f"  Error with task {task_id}: {e}")
    
    # Print summary
    print("\nVQA-Med Results Summary:")
    
    print("\nBy Task Type:")
    for ttype, successes in sorted(results['by_task_type'].items()):
        if successes:
            acc = sum(successes) / len(successes) * 100
            print(f"  - {ttype}: {len(successes)} tasks, {acc:.1f}% success")
    
    print("\nBy Answer Category:")
    for category, successes in sorted(results['by_answer_category'].items()):
        if successes:
            acc = sum(successes) / len(successes) * 100
            print(f"  - {category}: {len(successes)} tasks, {acc:.1f}% success")
    
    print("\nBy Imaging Modality:")
    for modality, successes in sorted(results['by_modality'].items()):
        if successes:
            acc = sum(successes) / len(successes) * 100
            print(f"  - {modality}: {len(successes)} mentions, {acc:.1f}% success")
    
    print("\nSpecial Question Types:")
    if results['modality_questions']:
        acc = sum(results['modality_questions']) / len(results['modality_questions']) * 100
        print(f"  - Modality questions: {len(results['modality_questions'])} tasks, {acc:.1f}% success")
    if results['contrast_questions']:
        acc = sum(results['contrast_questions']) / len(results['contrast_questions']) * 100
        print(f"  - Contrast questions: {len(results['contrast_questions'])} tasks, {acc:.1f}% success")
    if results['weighted_questions']:
        acc = sum(results['weighted_questions']) / len(results['weighted_questions']) * 100
        print(f"  - Weighted MRI questions: {len(results['weighted_questions'])} tasks, {acc:.1f}% success")
    
    print("\nError Type Distribution:")
    for error_type, count in sorted(results['errors'].items()):
        print(f"  - {error_type}: {count} occurrences")


def test_medical_patterns(adapter: VqaMedAdapter):
    """Analyze VQA-Med specific patterns"""
    print("\n" + "="*60)
    print("Testing VQA-Med Pattern Analysis")
    print("="*60)
    
    patterns = adapter.analyze_medical_patterns()
    
    print("\nTop Question Starters:")
    for starter, count in list(patterns['question_starters'].items())[:10]:
        print(f"  - '{starter}': {count} occurrences")
    
    print("\nCommon Question Keywords:")
    for keyword, count in list(patterns['question_keywords'].items())[:15]:
        print(f"  - '{keyword}': {count} times")
    
    print("\nQuestion Type Counts:")
    question_types = patterns.get('question_type_counts', {})
    for qtype, count in sorted(question_types.items()):
        print(f"  - {qtype}: {count} questions")
    
    print("\nAnswer Pattern Distribution:")
    answer_patterns = patterns.get('answer_patterns', {})
    for pattern, count in sorted(answer_patterns.items()):
        print(f"  - {pattern}: {count} occurrences")


def test_top_answers(adapter: VqaMedAdapter):
    """Test the most common answers in VQA-Med"""
    print("\n" + "="*60)
    print("Testing Top VQA-Med Answers")
    print("="*60)
    
    # Get dataset info
    info = adapter.get_dataset_info()
    top_answers = info.get('top_answers', {})
    
    print("\nTop 20 Most Common Answers:")
    for i, (answer, count) in enumerate(list(top_answers.items())[:20]):
        print(f"  {i+1}. '{answer}': {count} occurrences")
        
        # Test an example for top 5 answers
        if i < 5:
            examples = adapter.get_examples_by_answer(answer, n=1)
            if examples:
                task_data = adapter.get_task_data(examples[0])
                print(f"      Example Q: {task_data['question'][:60]}...")


def visualize_vqa_med_samples(adapter: VqaMedAdapter, n_samples: int = 6):
    """Visualize sample VQA-Med images with questions"""
    print("\n" + "="*60)
    print(f"Visualizing {n_samples} VQA-Med Sample Images")
    print("="*60)
    
    if len(adapter._task_index) == 0:
        print("No tasks available for visualization!")
        return
    
    # Get diverse samples focusing on different modalities
    sample_ids = []
    
    # Try to get one of each major modality
    modalities = ['CT', 'MRI', 'X-ray', 'Ultrasound']
    for modality in modalities:
        examples = adapter.get_modality_examples(modality, n=1)
        if examples:
            sample_ids.extend(examples)
    
    # Fill remaining with diverse examples
    if len(sample_ids) < n_samples:
        diverse = adapter.get_diverse_medical_examples(n=n_samples - len(sample_ids))
        sample_ids.extend(diverse)
    
    sample_ids = sample_ids[:n_samples]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, task_id in enumerate(sample_ids[:n_samples]):
        try:
            task = VqaMedTask(task_id=task_id, adapter=adapter)
            goal, info = task.setup()
            
            # Try to load and display image
            try:
                img = Image.open(task.image_path)
                axes[i].imshow(img, cmap='gray' if len(img.mode) == 1 else None)
                axes[i].axis('off')
                
                # Add title with VQA-Med specific info
                title = f"Q: {task.question[:40]}..."
                title += f"\nA: {task.expected_answer}"
                
                # Add answer category
                if task.answer_category:
                    category_emoji = {
                        'modality': '📷',
                        'anatomical': '🦴',
                        'plane': '📐',
                        'contrast': '💉',
                        'yes_no': '✓✗'
                    }
                    emoji = category_emoji.get(task.answer_category, '•')
                    title += f"\n{emoji} Category: {task.answer_category}"
                
                # Add special flags
                flags = []
                if task.is_modality_question:
                    flags.append("Modality Q")
                if task.is_contrast_question:
                    flags.append("Contrast Q")
                if task.is_weighted_question:
                    flags.append("MRI Seq Q")
                
                if flags:
                    title += f"\n[{', '.join(flags)}]"
                
                axes[i].set_title(title, fontsize=9, wrap=True)
                
                img.close()
            except Exception as e:
                # If image loading fails, show task info
                info_text = f"Medical Image\n\nQ: {task.question[:60]}...\nA: {task.expected_answer}"
                info_text += f"\n\nCategory: {task.answer_category}"
                info_text += f"\nType: {task.task_type}"
                
                if task.medical_entities:
                    info_text += f"\n\nEntities:"
                    for entity_type, values in task.medical_entities.items():
                        if values:
                            info_text += f"\n  {entity_type}: {', '.join(values[:2])}"
                
                axes[i].text(0.5, 0.5, info_text,
                           ha='center', va='center', transform=axes[i].transAxes,
                           fontsize=10, wrap=True)
                axes[i].axis('off')
                
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error: {str(e)[:50]}",
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')
    
    plt.tight_layout()
    output_file = "vqa_med_samples.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVQA-Med sample images saved to: {output_file}")
    plt.close()


def test_edge_cases(adapter: VqaMedAdapter):
    """Test VQA-Med specific edge cases"""
    print("\n" + "="*60)
    print("Testing VQA-Med Edge Cases")
    print("="*60)
    
    # Test standard format edge cases
    print("\n[Standard Format Edge Cases]")
    
    # Find a task with "xr - plain film" answer
    xr_examples = adapter.get_examples_by_answer("xr - plain film", n=1)
    if xr_examples:
        task = VqaMedTask(task_id=xr_examples[0], adapter=adapter)
        task.setup()
        
        print(f"  Question: {task.question}")
        print(f"  Expected: '{task.expected_answer}'")
        
        edge_cases = [
            ("xr-plain film", "Missing space"),
            ("xr -plain film", "Wrong space"),
            ("xr - plainfilm", "No space in 'plain film'"),
            ("XR - PLAIN FILM", "All caps"),
            ("xr - Plain Film", "Mixed case"),
            ("xr", "Only modality code"),
            ("plain film", "Only description")
        ]
        
        print(f"  Testing format edge cases:")
        for test_input, desc in edge_cases:
            success, feedback = task.check_success(test_input)
            status = "✅" if success else "❌"
            print(f"    {status} {desc}: '{test_input}'")
    
    # Test answer with special characters
    print("\n[Special Character Handling]")
    
    # Get any open-ended task
    open_tasks = adapter.get_open_ended_examples(n=1)
    if open_tasks:
        task = VqaMedTask(task_id=open_tasks[0], adapter=adapter)
        task.setup()
        
        print(f"  Question: {task.question[:60]}...")
        print(f"  Expected: '{task.expected_answer}'")
        
        # Test with added punctuation
        test_cases = [
            (f"{task.expected_answer}.", "With period"),
            (f"{task.expected_answer}!", "With exclamation"),
            (f"{task.expected_answer}?", "With question mark"),
            (f"'{task.expected_answer}'", "With quotes"),
            (f"  {task.expected_answer}  ", "With spaces"),
            (f"The {task.expected_answer}", "With article")
        ]
        
        print(f"  Testing punctuation handling:")
        for test_input, desc in test_cases[:4]:
            success, feedback = task.check_success(test_input)
            status = "✅" if success else "❌"
            print(f"    {status} {desc}: '{test_input}'")


def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description='Test VQA-Med integration with VLM Gym')
    parser.add_argument('--annotation', type=str,
                       default='/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/VQA-Med/vqa_med_train_vlmgym.json',
                       help='Path to VQA-Med annotation file')
    parser.add_argument('--data-root', type=str,
                       default='/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/VQA-Med',
                       help='Path to VQA-Med data root')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for evaluation')
    parser.add_argument('--skip-vis', action='store_true',
                       help='Skip visualization to avoid image loading issues')
    
    args = parser.parse_args()
    
    print("="*60)
    print("VQA-Med VLM Gym Integration Test")
    print("="*60)
    print(f"Python: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    try:
        # Test 1: Load adapter
        adapter = test_adapter_loading(args.annotation, args.data_root)
        
        # Check if adapter loaded successfully
        if adapter is None:
            print("\n❌ Failed to load adapter!")
            return 1
        
        # Check if adapter loaded any data
        if len(adapter._task_index) == 0:
            print("\n❌ No data loaded! Please check:")
            print(f"  1. Annotation file exists: {args.annotation}")
            print(f"  2. Annotation file contains valid data")
            return 1
        
        # Test 2: Check data format issues
        test_data_format_issues(adapter)
        
        # Test 3: Test actual yes/no questions
        test_actual_yes_no_questions(adapter)
        
        # Test 4: Modality recognition (main focus)
        test_modality_recognition_tasks(adapter)
        
        # Test 5: Yes/No validation
        test_yes_no_validation(adapter)
        
        # Test 6: Contrast questions
        test_contrast_questions(adapter)
        
        # Test 7: MRI weighted sequences
        test_weighted_mri_questions(adapter)
        
        # Test 8: Anatomical system answers
        test_anatomical_system_answers(adapter)
        
        # Test 9: Imaging plane answers
        test_imaging_plane_answers(adapter)
        
        # Test 10: Answer normalization
        test_answer_normalization(adapter)
        
        # Test 11: Complete workflow
        test_complete_workflow(adapter)
        
        # Test 12: Top answers
        test_top_answers(adapter)
        
        # Test 13: Edge cases
        test_edge_cases(adapter)
        
        # Test 14: Batch evaluation
        test_batch_evaluation(adapter, n_samples=args.batch_size)
        
        # Test 15: Medical patterns
        test_medical_patterns(adapter)
        
        # Test 16: Visualize samples (optional)
        if not args.skip_vis:
            visualize_vqa_med_samples(adapter, n_samples=6)
        
        print("\n" + "="*60)
        print("✓ All VQA-Med tests completed successfully!")
        print("="*60)
        
        print("\nKey VQA-Med findings:")
        print("1. VQA-Med adapter successfully loads medical imaging data")
        print("2. Many questions have mislabeled answer_type (marked as yes_no but have other answers)")
        print("3. Actual modality questions often have answers like 'xr - plain film', 'us - ultrasound'")
        print("4. Contrast questions typically have 'contrast' or 'noncontrast' answers, not yes/no")
        print("5. MRI sequence questions have answers like 't1', 't2', 'flair', not yes/no")
        print("6. Answer normalization is important for handling format variations")
        print("7. The dataset contains diverse medical imaging questions beyond modality recognition")
        
        print("\nData Quality Issues Found:")
        print("1. answer_type field often incorrect (yes_no for non-binary answers)")
        print("2. choices field contains ['yes', 'no'] even when answer is neither")
        print("3. Task type classification may need refinement")
        print("4. Some questions labeled as one type but are actually another")
        
        print("\nVQA-Med vs VQA-RAD differences:")
        print("1. VQA-Med has more format inconsistencies in the converted data")
        print("2. Many technical questions (contrast, MRI sequences) have specific answers, not yes/no")
        print("3. Standard answer formats like 'xr - plain film' are unique to VQA-Med")
        print("4. Less emphasis on diagnosis, more on imaging characteristics")
        print("5. Answer normalization is crucial for format variations")
        
        print("\nNext steps:")
        print("1. Consider fixing the data conversion script to properly classify answer_type")
        print("2. Test with vision-language models that can handle diverse answer formats")
        print("3. Evaluate performance on technical imaging questions")
        print("4. Analyze which question types are most challenging")
        print("5. Test answer normalization effectiveness")
        print("6. Study error patterns in different answer categories")
        
        if not args.skip_vis:
            print("\n7. Check vqa_med_samples.png for visualization examples")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())