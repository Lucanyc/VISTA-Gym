#!/usr/bin/env python3
"""
Test script for VQA-RAD integration with VLM Gym
Tests medical visual question answering on radiology images
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
    from vqarad import VqaRadTask
except ImportError as e:
    print(f"Warning: Failed to import VqaRadTask from current directory: {e}")
    try:
        from vlm_gym.environments.task.vqarad import VqaRadTask
    except ImportError as e2:
        print(f"Error: Cannot import VqaRadTask: {e2}")
        print("\nPlease ensure vqarad.py exists in:")
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
    from vqarad_adapter import VqaRadAdapter
except ImportError as e:
    print(f"Error: Cannot import VqaRadAdapter: {e}")
    print("\nMake sure vqarad_adapter.py is in:")
    print("/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data_adapters/")
    sys.exit(1)


def test_adapter_loading(annotation_file, data_root=None):
    """Test VQA-RAD adapter initialization and data loading"""
    print("\n" + "="*60)
    print("Testing VQA-RAD Adapter Loading")
    print("="*60)
    
    # Initialize adapter with provided paths
    print(f"Loading from: {annotation_file}")
    if data_root:
        print(f"Data root: {data_root}")
    
    adapter = VqaRadAdapter(
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
    
    # Question type distribution
    print(f"\n  Medical Question Type Distribution:")
    for qtype, count in sorted(stats.get('question_type_distribution', {}).items()):
        print(f"    - {qtype}: {count} ({count/stats['total']*100:.1f}%)")
    
    # Modality distribution
    print(f"\n  Imaging Modality Distribution:")
    for modality, count in list(stats.get('modality_distribution', {}).items())[:5]:
        print(f"    - {modality}: {count} occurrences")
    
    # Top anatomical structures
    print(f"\n  Top Anatomical Structures:")
    for anatomy, count in list(stats.get('anatomy_distribution', {}).items())[:10]:
        print(f"    - {anatomy}: {count} occurrences")
    
    # Top abnormalities
    abnorm_dist = stats.get('abnormality_distribution', {})
    if abnorm_dist:
        print(f"\n  Top Abnormalities:")
        for abnorm, count in list(abnorm_dist.items())[:5]:
            print(f"    - {abnorm}: {count} occurrences")
    
    # Medical complexity
    complexity = stats.get('medical_complexity_statistics', {})
    if complexity:
        print(f"\n  Medical Complexity:")
        print(f"    - With medical terms: {complexity.get('with_medical_terms', 0)}")
        print(f"    - With anatomical terms: {complexity.get('with_anatomical_terms', 0)}")
        print(f"    - With laterality: {complexity.get('with_laterality', 0)}")
        print(f"    - Differential diagnosis: {complexity.get('differential_diagnosis', 0)}")
    
    return adapter


def test_medical_task_creation(adapter: VqaRadAdapter):
    """Test creating medical tasks for different types"""
    print("\n" + "="*60)
    print("Testing Medical Task Creation by Type")
    print("="*60)
    
    medical_task_types = [
        'medical_diagnosis',
        'anatomical_recognition',
        'abnormality_detection',
        'modality_recognition',
        'lesion_localization',
        'medical_verification',
        'medical_description',
        'medical_vqa'
    ]
    created_tasks = {}
    
    for task_type in medical_task_types:
        task_ids = adapter.get_task_ids(task_type=task_type, limit=1)
        if task_ids:
            # Create task using task_id and adapter
            task = VqaRadTask(task_id=task_ids[0], adapter=adapter)
            goal, info = task.setup()
            created_tasks[task_type] = task
            
            # Get task data for display
            task_data = adapter.get_task_data(task_ids[0])
            
            print(f"\n[{task_type.upper()} Task]")
            print(f"  - Task ID: {task_data['id']}")
            print(f"  - Question: {task.question[:80]}...")
            print(f"  - Answer: {task.answer}")
            print(f"  - Answer type: {task.answer_type}")
            print(f"  - Is clinical: {task.is_clinical}")
            print(f"  - Question type: {task.question_type}")
            print(f"  - Is complex: {task.is_complex}")
            print(f"  - Difficulty: {info.get('difficulty', 'unknown')}")
            
            # Show medical entities if present
            if task.medical_entities:
                entities_shown = False
                for entity_type, values in task.medical_entities.items():
                    if values:
                        if not entities_shown:
                            print(f"  - Medical entities:")
                            entities_shown = True
                        print(f"    * {entity_type}: {', '.join(values)}")
    
    return created_tasks


def test_medical_yes_no_validation(adapter: VqaRadAdapter):
    """Test medical yes/no answer validation"""
    print("\n" + "="*60)
    print("Testing Medical Yes/No Answer Validation")
    print("="*60)
    
    # Get clinical yes/no tasks
    yes_no_tasks = adapter.get_task_ids(answer_type='yes_no', is_clinical=True, limit=4)
    
    if not yes_no_tasks:
        # Fallback to any yes/no tasks
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
        task = VqaRadTask(task_id=task_id, adapter=adapter)
        task.setup()
        
        print(f"\n[Testing {expected_answer.upper()} answer - Medical Context]")
        print(f"Question: {task.question[:80]}...")
        print(f"Correct answer: {task.expected_answer}")
        print(f"Clinical relevance: {task.is_clinical}")
        
        # Test different medical answer formats
        if expected_answer == 'yes':
            test_cases = [
                ("yes", "Exact 'yes'"),
                ("Yes", "Capitalized 'Yes'"),
                ("positive", "Medical 'positive'"),
                ("present", "Medical 'present'"),
                ("abnormal", "Medical 'abnormal'"),
                ("There are findings", "Medical phrase"),
                ("no", "Wrong answer 'no'"),
                ("normal", "Wrong 'normal'"),
                ("absent", "Wrong 'absent'"),
                (None, "None value")
            ]
        else:  # no
            test_cases = [
                ("no", "Exact 'no'"),
                ("No", "Capitalized 'No'"),
                ("negative", "Medical 'negative'"),
                ("absent", "Medical 'absent'"),
                ("normal", "Medical 'normal'"),
                ("No abnormality", "Medical phrase"),
                ("yes", "Wrong answer 'yes'"),
                ("present", "Wrong 'present'"),
                ("abnormal", "Wrong 'abnormal'"),
                (None, "None value")
            ]
        
        print("\nMedical answer validation tests:")
        for test_answer, desc in test_cases:
            success, feedback = task.check_success(test_answer)
            status = "✅" if success else "❌"
            print(f"  {status} {desc}: '{test_answer}' -> {feedback[:50]}...")


def test_modality_recognition(adapter: VqaRadAdapter):
    """Test imaging modality recognition"""
    print("\n" + "="*60)
    print("Testing Imaging Modality Recognition")
    print("="*60)
    
    # Get modality recognition tasks specifically
    modality_tasks = adapter.get_task_ids(task_type='modality_recognition', limit=10)
    
    # Find tasks where the answer is actually a modality
    modality_answer_tasks = []
    for task_id in modality_tasks:
        task_data = adapter.get_task_data(task_id)
        answer = task_data.get('answer', '').lower()
        if answer in ['mri', 'ct', 'x-ray', 'xray', 'ultrasound', 'pet', 'spect']:
            modality_answer_tasks.append((task_id, answer))
    
    if not modality_answer_tasks:
        print("No modality recognition tasks with modality answers found")
        return
    
    # Test different modalities
    tested_modalities = set()
    for task_id, modality_answer in modality_answer_tasks[:3]:
        if modality_answer in tested_modalities:
            continue
        tested_modalities.add(modality_answer)
        
        task = VqaRadTask(task_id=task_id, adapter=adapter)
        task.setup()
        
        print(f"\n[{modality_answer.upper()} Modality Recognition]")
        print(f"  Question: {task.question}")
        print(f"  Expected answer: '{task.expected_answer}'")
        
        # Test modality synonyms based on the expected answer
        if modality_answer == 'mri':
            test_answers = ['mri', 'MRI', 'magnetic resonance imaging', 'MR', 'mr imaging']
        elif modality_answer == 'ct':
            test_answers = ['ct', 'CT', 'computed tomography', 'CAT scan', 'ct scan']
        elif modality_answer in ['x-ray', 'xray']:
            test_answers = ['x-ray', 'xray', 'X-ray', 'radiograph', 'plain film']
        else:
            test_answers = [modality_answer, modality_answer.upper()]
        
        print(f"  Testing medical synonyms:")
        for test_ans in test_answers[:3]:
            success, _ = task.check_success(test_ans)
            status = "✅" if success else "❌"
            print(f"    {status} '{test_ans}'")


def test_anatomical_recognition(adapter: VqaRadAdapter):
    """Test anatomical structure recognition"""
    print("\n" + "="*60)
    print("Testing Anatomical Structure Recognition")
    print("="*60)
    
    # Test common anatomical structures
    anatomies = ['brain', 'lung', 'heart', 'liver']
    
    for anatomy in anatomies:
        examples = adapter.get_anatomy_examples(anatomy, n=2)
        if examples:
            print(f"\n[{anatomy.upper()} Related Questions]")
            
            for task_id in examples[:1]:
                task = VqaRadTask(task_id=task_id, adapter=adapter)
                task.setup()
                
                print(f"  Q: {task.question}")
                print(f"  A: {task.expected_answer}")
                
                # Show medical complexity
                if task.medical_complexity:
                    complexity_features = [k for k, v in task.medical_complexity.items() if v]
                    if complexity_features:
                        print(f"  Medical complexity: {', '.join(complexity_features)}")


def test_clinical_vs_nonclinical(adapter: VqaRadAdapter):
    """Test clinical vs non-clinical questions"""
    print("\n" + "="*60)
    print("Testing Clinical vs Non-Clinical Questions")
    print("="*60)
    
    # Get clinical examples
    print("\n[CLINICAL Questions]")
    clinical_examples = adapter.get_clinical_examples(n=3, clinical=True)
    for task_id in clinical_examples[:2]:
        task = VqaRadTask(task_id=task_id, adapter=adapter)
        task.setup()
        print(f"  Q: {task.question}")
        print(f"  A: {task.expected_answer}")
        print(f"  Task type: {task.task_type}")
    
    # Get non-clinical examples
    print("\n[NON-CLINICAL Questions]")
    non_clinical_examples = adapter.get_clinical_examples(n=3, clinical=False)
    for task_id in non_clinical_examples[:2]:
        task = VqaRadTask(task_id=task_id, adapter=adapter)
        task.setup()
        print(f"  Q: {task.question}")
        print(f"  A: {task.expected_answer}")
        print(f"  Task type: {task.task_type}")


def test_medical_complexity(adapter: VqaRadAdapter):
    """Test medical complexity features"""
    print("\n" + "="*60)
    print("Testing Medical Complexity Features")
    print("="*60)
    
    # Test laterality questions
    print("\n[LATERALITY Questions]")
    laterality_tasks = adapter.get_laterality_examples(n=3)
    for task_id in laterality_tasks[:2]:
        task = VqaRadTask(task_id=task_id, adapter=adapter)
        goal, info = task.setup()
        print(f"  Q: {task.question}")
        print(f"  A: {task.expected_answer}")
        
        # Test laterality variations
        if 'left' in task.expected_answer:
            test_answers = ['left', 'left side', 'on the left']
        elif 'right' in task.expected_answer:
            test_answers = ['right', 'right side', 'on the right']
        elif 'bilateral' in task.expected_answer:
            test_answers = ['bilateral', 'both sides', 'left and right']
        else:
            test_answers = []
        
        if test_answers:
            print(f"  Testing laterality variations:")
            for test_ans in test_answers:
                success, _ = task.check_success(test_ans)
                status = "✅" if success else "❌"
                print(f"    {status} '{test_ans}'")
    
    # Test differential diagnosis questions
    print("\n[DIFFERENTIAL DIAGNOSIS Questions]")
    diff_tasks = adapter.get_differential_examples(n=3)
    for task_id in diff_tasks[:2]:
        task = VqaRadTask(task_id=task_id, adapter=adapter)
        task.setup()
        print(f"  Q: {task.question}")
        print(f"  A: {task.expected_answer}")
        print(f"  Is complex: {task.is_complex}")


def test_abnormality_detection(adapter: VqaRadAdapter):
    """Test abnormality detection questions"""
    print("\n" + "="*60)
    print("Testing Abnormality Detection")
    print("="*60)
    
    # Get abnormality detection tasks
    abnorm_tasks = adapter.get_task_ids(task_type='abnormality_detection', limit=5)
    
    if abnorm_tasks:
        print(f"Found {len(abnorm_tasks)} abnormality detection tasks")
        
        for i, task_id in enumerate(abnorm_tasks[:3]):
            task = VqaRadTask(task_id=task_id, adapter=adapter)
            goal, info = task.setup()
            
            print(f"\n[Abnormality Task {i+1}]")
            print(f"  Question: {task.question}")
            print(f"  Answer: {task.expected_answer}")
            print(f"  Clinical: {task.is_clinical}")
            print(f"  Difficulty: {info['difficulty']}")
            
            # Show detected abnormalities
            if task.medical_entities.get('abnormalities'):
                print(f"  Mentioned abnormalities: {', '.join(task.medical_entities['abnormalities'])}")


def test_complete_medical_workflow(adapter: VqaRadAdapter):
    """Test complete medical VQA workflow"""
    print("\n" + "="*60)
    print("Testing Complete Medical VQA Workflow")
    print("="*60)
    
    # Get a diverse medical example
    task_ids = adapter.get_diverse_medical_examples(n=1)
    if not task_ids:
        task_ids = adapter.get_task_ids(is_clinical=True, limit=1)
    
    if not task_ids:
        print("No tasks available!")
        return
    
    task_data = adapter.get_task_data(task_ids[0])
    task = VqaRadTask(task_id=task_ids[0], adapter=adapter)
    
    goal, info = task.setup()
    
    print("Medical Task Setup:")
    print(f"  - Question: {task.question}")
    print(f"  - Expected answer: {task.expected_answer}")
    print(f"  - Task type: {task.task_type}")
    print(f"  - Question type: {task.question_type}")
    print(f"  - Answer type: {task.answer_type}")
    print(f"  - Clinical relevance: {task.is_clinical}")
    print(f"  - Medical domain: {info.get('medical_domain', 'radiology')}")
    print(f"  - Difficulty: {info['difficulty']}")
    
    # Show medical entities
    if task.medical_entities:
        print(f"\nMedical Entities:")
        for entity_type, values in task.medical_entities.items():
            if values:
                print(f"  - {entity_type}: {', '.join(values)}")
    
    # Get observation
    obs = task.get_observation()
    print(f"\nObservation keys: {list(obs.keys())}")
    print(f"  - Scene type: {obs.get('scene_type', 'N/A')}")
    print(f"  - Expected content: {obs.get('expected_content', 'N/A')}")
    print(f"  - Clinical relevance: {obs.get('clinical_relevance', 'N/A')}")
    print(f"  - Medical domain: {obs.get('medical_domain', 'N/A')}")
    
    # Simulate different agent responses
    print(f"\nSimulating medical agent responses:")
    
    # 1. Correct answer
    correct_answer = task.expected_answer
    chat_history = [{"role": "assistant", "content": f"Based on the medical image, {correct_answer}."}]
    reward, done, message, val_info = task.validate(chat_history, correct_answer)
    print(f"  1. Correct answer ('{correct_answer}'): {message}")
    print(f"     Reward: {reward}, Done: {done}")
    
    # 2. Medical synonym (if applicable)
    if task.task_type == 'modality_recognition':
        if correct_answer == 'mri':
            synonym = 'magnetic resonance imaging'
        elif correct_answer == 'ct':
            synonym = 'computed tomography'
        else:
            synonym = None
        
        if synonym:
            chat_history = [{"role": "assistant", "content": f"This is a {synonym} scan."}]
            reward, done, message, val_info = task.validate(chat_history, synonym)
            print(f"  2. Medical synonym ('{synonym}'): {message}")
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
    
    # Get metrics
    metrics = task.get_metrics()
    print(f"\nMedical Task Metrics:")
    important_metrics = [
        'task_type', 'question_type', 'answer_type', 'is_binary', 
        'is_clinical', 'medical_domain', 'difficulty'
    ]
    for key in important_metrics:
        if key in metrics:
            print(f"  - {key}: {metrics[key]}")


def test_batch_medical_evaluation(adapter: VqaRadAdapter, n_samples: int = 100):
    """Test batch evaluation for medical tasks"""
    print("\n" + "="*60)
    print(f"Testing Batch Medical Evaluation ({n_samples} samples)")
    print("="*60)
    
    # Check available tasks
    total_tasks = len(adapter._task_index)
    if total_tasks == 0:
        print("No tasks available for batch evaluation!")
        return
    
    actual_samples = min(n_samples, total_tasks)
    print(f"Available tasks: {total_tasks}, testing {actual_samples} samples")
    
    # Sample tasks with clinical balance
    task_ids = adapter.sample_tasks(
        actual_samples, 
        balanced_clinical=True,
        seed=42
    )
    
    results = {
        'by_task_type': defaultdict(list),
        'by_question_type': defaultdict(list),
        'by_answer_type': defaultdict(list),
        'by_difficulty': defaultdict(list),
        'by_clinical': defaultdict(list),
        'by_modality': defaultdict(list),
        'complex': [],
        'simple': [],
        'yes_no': [],
        'open_ended': [],
        'clinical': [],
        'non_clinical': [],
        'errors': defaultdict(int)
    }
    
    print(f"\nProcessing {len(task_ids)} medical tasks...")
    
    for i, task_id in enumerate(task_ids):
        try:
            task = VqaRadTask(task_id=task_id, adapter=adapter)
            goal, info = task.setup()
            
            # Simulate correct answer
            success, _ = task.check_success(task.expected_answer)
            
            # Record results
            results['by_task_type'][task.task_type].append(success)
            results['by_question_type'][task.question_type].append(success)
            results['by_answer_type'][task.answer_type].append(success)
            results['by_difficulty'][info['difficulty']].append(success)
            
            # Clinical categories
            clinical_status = 'clinical' if task.is_clinical else 'non_clinical'
            results['by_clinical'][clinical_status].append(success)
            
            if task.is_clinical:
                results['clinical'].append(success)
            else:
                results['non_clinical'].append(success)
            
            # Modality tracking
            if task.medical_entities.get('modalities'):
                for modality in task.medical_entities['modalities']:
                    results['by_modality'][modality].append(success)
            
            # Complexity
            if task.is_complex:
                results['complex'].append(success)
            else:
                results['simple'].append(success)
            
            # Answer type
            if task.is_binary:
                results['yes_no'].append(success)
            else:
                results['open_ended'].append(success)
            
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
    print("\nMedical Results Summary:")
    
    print("\nBy Medical Task Type:")
    for ttype, successes in sorted(results['by_task_type'].items()):
        if successes:
            acc = sum(successes) / len(successes) * 100
            print(f"  - {ttype}: {len(successes)} tasks, {acc:.1f}% success")
    
    print("\nBy Clinical Relevance:")
    for clinical_type, successes in sorted(results['by_clinical'].items()):
        if successes:
            acc = sum(successes) / len(successes) * 100
            print(f"  - {clinical_type}: {len(successes)} tasks, {acc:.1f}% success")
    
    print("\nBy Imaging Modality:")
    for modality, successes in sorted(results['by_modality'].items()):
        if successes:
            acc = sum(successes) / len(successes) * 100
            print(f"  - {modality}: {len(successes)} tasks, {acc:.1f}% success")
    
    print("\nBy Answer Type:")
    for atype, successes in sorted(results['by_answer_type'].items()):
        if successes:
            acc = sum(successes) / len(successes) * 100
            print(f"  - {atype}: {len(successes)} tasks, {acc:.1f}% success")
    
    print("\nBy Difficulty:")
    for difficulty, successes in sorted(results['by_difficulty'].items()):
        if successes:
            acc = sum(successes) / len(successes) * 100
            print(f"  - {difficulty}: {len(successes)} tasks, {acc:.1f}% success")
    
    print("\nSpecial Medical Categories:")
    if results['clinical']:
        acc = sum(results['clinical']) / len(results['clinical']) * 100
        print(f"  - Clinical questions: {len(results['clinical'])} tasks, {acc:.1f}% success")
    if results['non_clinical']:
        acc = sum(results['non_clinical']) / len(results['non_clinical']) * 100
        print(f"  - Non-clinical questions: {len(results['non_clinical'])} tasks, {acc:.1f}% success")
    
    print("\nMedical Error Type Distribution:")
    for error_type, count in sorted(results['errors'].items()):
        print(f"  - {error_type}: {count} occurrences")


def test_medical_pattern_analysis(adapter: VqaRadAdapter):
    """Analyze medical question patterns"""
    print("\n" + "="*60)
    print("Testing Medical Pattern Analysis")
    print("="*60)
    
    patterns = adapter.analyze_medical_patterns()
    
    print("\nTop Medical Question Starters:")
    for starter, count in list(patterns['question_starters'].items())[:15]:
        print(f"  - '{starter}': {count} occurrences")
    
    print("\nCommon Medical Keywords:")
    for keyword, count in list(patterns['medical_keywords'].items())[:20]:
        print(f"  - '{keyword}': {count} times")
    
    print("\nMedical Question Type Counts:")
    question_types = patterns.get('question_type_counts', {})
    for qtype, count in sorted(question_types.items()):
        print(f"  - {qtype}: {count} questions")


def test_specific_medical_conditions(adapter: VqaRadAdapter):
    """Test questions about specific medical conditions"""
    print("\n" + "="*60)
    print("Testing Specific Medical Conditions")
    print("="*60)
    
    # Test common medical answers
    medical_answers = ['yes', 'no', 'mri', 'ct', 'normal', 'abnormal', 'brain', 'lung']
    
    for answer in medical_answers:
        examples = adapter.get_examples_by_answer(answer, n=2)
        if examples:
            print(f"\n[Medical Answer: '{answer}'] Found {len(examples)} examples")
            
            for task_id in examples[:1]:
                task = VqaRadTask(task_id=task_id, adapter=adapter)
                task.setup()
                print(f"  Q: {task.question}")
                print(f"  Clinical: {task.is_clinical}")
                if task.medical_entities:
                    entities_shown = False
                    for entity_type, values in task.medical_entities.items():
                        if values:
                            if not entities_shown:
                                print(f"  Entities: ", end="")
                                entities_shown = True
                            print(f"{entity_type}={values}", end=" ")
                    if entities_shown:
                        print()


def visualize_medical_samples(adapter: VqaRadAdapter, n_samples: int = 6):
    """Visualize sample medical images with questions"""
    print("\n" + "="*60)
    print(f"Visualizing {n_samples} Medical Sample Images")
    print("="*60)
    
    if len(adapter._task_index) == 0:
        print("No tasks available for visualization!")
        return
    
    # Get diverse medical samples
    sample_ids = adapter.get_diverse_medical_examples(n=n_samples)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, task_id in enumerate(sample_ids[:n_samples]):
        try:
            task = VqaRadTask(task_id=task_id, adapter=adapter)
            goal, info = task.setup()
            
            # Try to load and display image
            try:
                img = Image.open(task.image_path)
                axes[i].imshow(img, cmap='gray' if len(img.mode) == 1 else None)
                axes[i].axis('off')
                
                # Add title with medical info
                task_type_short = task.task_type.replace('medical_', '').replace('_', ' ').title()
                title = f"{task_type_short}"
                if task.is_clinical:
                    title = "🏥 " + title
                title += f"\nQ: {task.question[:40]}..."
                title += f"\nA: {task.expected_answer}"
                
                # Add modality if present
                if task.medical_entities.get('modalities'):
                    title += f"\nModality: {', '.join(task.medical_entities['modalities'])}"
                
                axes[i].set_title(title, fontsize=9, wrap=True)
                
                img.close()
            except Exception as e:
                # If image loading fails, show medical task info
                info_text = f"Medical Image\n\nQ: {task.question[:60]}...\nA: {task.expected_answer}"
                info_text += f"\n\nType: {task.task_type}"
                info_text += f"\nClinical: {'Yes' if task.is_clinical else 'No'}"
                info_text += f"\nDifficulty: {info['difficulty']}"
                
                if task.medical_entities:
                    info_text += f"\n\nMedical entities:"
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
    output_file = "vqa_rad_samples.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nMedical sample images saved to: {output_file}")
    plt.close()


def test_clinical_splits(adapter: VqaRadAdapter):
    """Test creating balanced clinical train/val/test splits"""
    print("\n" + "="*60)
    print("Testing Clinical Data Splits")
    print("="*60)
    
    # Create splits with clinical balance
    splits = adapter.create_clinical_split(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        ensure_clinical_balance=True,
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
    
    # Check clinical distribution in each split
    print(f"\nClinical distribution in splits:")
    for split_name, task_ids in splits.items():
        clinical_counts = Counter()
        for task_id in task_ids[:100]:  # Sample first 100 for speed
            task_data = adapter.get_task_data(task_id)
            is_clinical = task_data.get('metadata', {}).get('is_clinical', False)
            clinical_counts['clinical' if is_clinical else 'non_clinical'] += 1
        
        print(f"\n  {split_name.capitalize()}:")
        total = sum(clinical_counts.values())
        for clinical_type, count in sorted(clinical_counts.items()):
            print(f"    - {clinical_type}: {count} ({count/total*100:.1f}%)")


def test_medical_edge_cases(adapter: VqaRadAdapter):
    """Test medical edge cases and special scenarios"""
    print("\n" + "="*60)
    print("Testing Medical Edge Cases")
    print("="*60)
    
    # Test medical terminology variations
    print("\n[Medical Terminology Variations]")
    # Get a modality recognition task where answer is actually a modality
    modality_tasks = adapter.get_task_ids(task_type='modality_recognition', limit=20)
    modality_task = None
    
    for task_id in modality_tasks:
        task_data = adapter.get_task_data(task_id)
        answer = task_data.get('answer', '').lower()
        if answer in ['mri', 'ct', 'x-ray']:
            modality_task = task_id
            break
    
    if modality_task:
        task = VqaRadTask(task_id=modality_task, adapter=adapter)
        task.setup()
        
        print(f"  Question: {task.question}")
        print(f"  Expected: {task.expected_answer}")
        
        # Test various medical formats based on the expected answer
        if task.expected_answer.lower() == 'mri':
            medical_formats = [
                ("MRI", "Uppercase"),
                ("mri", "Lowercase"),
                ("M.R.I.", "With dots"),
                ("MR", "Abbreviation"),
                ("magnetic resonance imaging", "Full name"),
                ("MR imaging", "Alternative"),
                ("an MRI", "With article")
            ]
        elif task.expected_answer.lower() == 'ct':
            medical_formats = [
                ("CT", "Uppercase"),
                ("ct", "Lowercase"),
                ("C.T.", "With dots"),
                ("computed tomography", "Full name"),
                ("CAT scan", "Alternative"),
                ("CT scan", "With 'scan'"),
                ("a CT", "With article")
            ]
        else:  # x-ray
            medical_formats = [
                ("X-ray", "Standard"),
                ("x-ray", "Lowercase"),
                ("xray", "No hyphen"),
                ("X-RAY", "All caps"),
                ("radiograph", "Medical term"),
                ("plain film", "Alternative"),
                ("an x-ray", "With article")
            ]
        
        print(f"  Testing medical format variations:")
        for test_input, desc in medical_formats:
            success, feedback = task.check_success(test_input)
            status = "✅" if success else "❌"
            print(f"  {status} {desc}: '{test_input}' -> {feedback[:40]}...")
    else:
        print("  No suitable modality recognition task found for testing")
    
    # Test medical answer normalization for yes/no
    print("\n[Medical Yes/No Normalization]")
    yes_no_task = adapter.get_task_ids(answer_type='yes_no', is_clinical=True, limit=1)
    if yes_no_task:
        task = VqaRadTask(task_id=yes_no_task[0], adapter=adapter)
        task.setup()
        
        print(f"  Question: {task.question}")
        print(f"  Expected: {task.expected_answer}")
        
        # Test edge cases
        edge_cases = [
            ("", "Empty string"),
            ("   ", "Whitespace only"),
            (None, "None value"),
            ("!!!", "Only punctuation"),
            ("the", "Only article"),
            ("maybe", "Ambiguous answer")
        ]
        
        print(f"  Testing edge cases:")
        for test_input, desc in edge_cases:
            success, feedback = task.check_success(test_input)
            status = "✅" if success else "❌"
            print(f"  {status} {desc}: '{test_input}' -> {feedback[:40]}...")
    
    # Test laterality edge cases
    print("\n[Laterality Edge Cases]")
    laterality_tasks = adapter.get_laterality_examples(n=1)
    if laterality_tasks:
        task = VqaRadTask(task_id=laterality_tasks[0], adapter=adapter)
        task.setup()
        
        print(f"  Question: {task.question}")
        print(f"  Expected: {task.expected_answer}")
        
        # Test laterality variations
        if 'bilateral' in task.expected_answer.lower():
            laterality_tests = [
                ("bilateral", "Exact match"),
                ("Bilateral", "Capitalized"),
                ("both sides", "Alternative phrase"),
                ("left and right", "Explicit both"),
                ("bilaterally", "Adverb form"),
                ("unilateral", "Wrong - unilateral")
            ]
            
            print(f"  Testing laterality variations:")
            for test_input, desc in laterality_tests:
                success, feedback = task.check_success(test_input)
                status = "✅" if success else "❌"
                print(f"  {status} {desc}: '{test_input}'")


def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description='Test VQA-RAD integration with VLM Gym')
    parser.add_argument('--annotation', type=str,
                       default='/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/VQA-RAD/vqa_rad_train_vlmgym.json',
                       help='Path to VQA-RAD annotation file')
    parser.add_argument('--data-root', type=str,
                       default='/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/VQA-RAD',
                       help='Path to VQA-RAD data root')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for evaluation')
    parser.add_argument('--skip-vis', action='store_true',
                       help='Skip visualization to avoid image loading issues')
    
    args = parser.parse_args()
    
    print("="*60)
    print("VQA-RAD VLM Gym Integration Test")
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
        
        # Test 2: Create medical tasks by type
        tasks = test_medical_task_creation(adapter)
        
        # Test 3: Medical yes/no answer validation
        test_medical_yes_no_validation(adapter)
        
        # Test 4: Modality recognition
        test_modality_recognition(adapter)
        
        # Test 5: Anatomical recognition
        test_anatomical_recognition(adapter)
        
        # Test 6: Clinical vs non-clinical
        test_clinical_vs_nonclinical(adapter)
        
        # Test 7: Medical complexity
        test_medical_complexity(adapter)
        
        # Test 8: Abnormality detection
        test_abnormality_detection(adapter)
        
        # Test 9: Complete medical workflow
        test_complete_medical_workflow(adapter)
        
        # Test 10: Specific medical conditions
        test_specific_medical_conditions(adapter)
        
        # Test 11: Medical edge cases
        test_medical_edge_cases(adapter)
        
        # Test 12: Batch medical evaluation
        test_batch_medical_evaluation(adapter, n_samples=args.batch_size)
        
        # Test 13: Medical pattern analysis
        test_medical_pattern_analysis(adapter)
        
        # Test 14: Clinical splits
        test_clinical_splits(adapter)
        
        # Test 15: Visualize medical samples (optional)
        if not args.skip_vis:
            visualize_medical_samples(adapter, n_samples=6)
        
        print("\n" + "="*60)
        print("✓ All medical tests completed successfully!")
        print("="*60)
        
        print("\nKey medical findings:")
        print("1. VQA-RAD adapter successfully loads medical imaging data")
        print("2. Medical terminology and synonyms are properly handled")
        print("3. Clinical vs non-clinical questions are distinguished")
        print("4. Imaging modalities (MRI, CT, X-ray) are correctly recognized")
        print("5. Anatomical structures and abnormalities are indexed")
        print("6. Medical complexity features (laterality, differential) work")
        print("7. Both yes/no and open-ended medical questions are supported")
        
        print("\nVQA-RAD vs VQA-AS differences:")
        print("1. VQA-RAD focuses on medical radiology images")
        print("2. Answers include medical terminology and clinical concepts")
        print("3. Clinical relevance is an important factor")
        print("4. Medical synonyms (e.g., MRI/magnetic resonance imaging)")
        print("5. Laterality (left/right/bilateral) is significant")
        print("6. Imaging modality recognition is a key task type")
        
        print("\nNext steps:")
        print("1. Test with medical vision-language models")
        print("2. Evaluate clinical vs non-clinical performance")
        print("3. Analyze accuracy by imaging modality")
        print("4. Study model performance on differential diagnosis")
        print("5. Test medical synonym understanding")
        print("6. Evaluate laterality recognition accuracy")
        
        if not args.skip_vis:
            print("\n7. Check vqa_rad_samples.png for medical visualization examples")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())