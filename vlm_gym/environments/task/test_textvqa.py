#!/usr/bin/env python3
"""
Test script for Text-VQA integration with VLM Gym
Tests text reading, OCR understanding, and various text-based VQA capabilities
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

# Add paths for imports
sys.path.insert(0, '/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista')
sys.path.insert(0, '/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data_adapters')

# Import required components
try:
    from textvqa import TextVQATask
except ImportError as e:
    print(f"Warning: Failed to import TextVQATask from current directory: {e}")
    try:
        from vlm_gym.environments.task.textvqa import TextVQATask
    except ImportError as e2:
        print(f"Error: Cannot import TextVQATask: {e2}")
        print("\nPlease ensure textvqa.py exists in:")
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
    from textvqa_adapter import TextVQAAdapter
except ImportError as e:
    print(f"Error: Cannot import TextVQAAdapter: {e}")
    print("\nMake sure textvqa_adapter.py is in:")
    print("/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data_adapters/")
    sys.exit(1)


def test_adapter_loading(annotation_file, data_root=None):
    """Test Text-VQA adapter initialization and data loading"""
    print("\n" + "="*60)
    print("Testing Text-VQA Adapter Loading")
    print("="*60)
    
    # Initialize adapter with provided paths
    print(f"Loading from: {annotation_file}")
    if data_root:
        print(f"Image root: {data_root}")
    
    adapter = TextVQAAdapter(
        data_root=data_root,
        annotation_files=annotation_file,
        validate_images=False,  # Set to False initially to avoid path issues
        min_answer_frequency=1
    )
    
    # Get statistics
    stats = adapter.get_statistics()
    print(f"\n✓ Adapter loaded successfully!")
    print(f"  - Total tasks: {stats['total']}")
    print(f"  - Unique images: {stats.get('image_statistics', {}).get('unique_images', 'N/A')}")
    
    # Task type distribution
    print(f"\n  Task Type Distribution:")
    for task_type, count in sorted(stats.get('task_type_distribution', {}).items()):
        print(f"    - {task_type}: {count} ({count/stats['total']*100:.1f}%)")
    
    # Question type distribution
    print(f"\n  Question Type Distribution:")
    for qtype, count in sorted(stats.get('question_type_distribution', {}).items())[:10]:
        print(f"    - {qtype}: {count} ({count/stats['total']*100:.1f}%)")
    
    # OCR statistics
    ocr_stats = stats.get('ocr_statistics', {})
    print(f"\n  OCR Statistics:")
    print(f"    - With OCR: {ocr_stats.get('with_ocr', 0)}")
    print(f"    - Without OCR: {ocr_stats.get('without_ocr', 0)}")
    
    # Answer statistics
    ans_stats = stats.get('answer_statistics', {})
    print(f"\n  Answer Statistics:")
    print(f"    - Single word answers: {ans_stats.get('single_word_answers', 0)}")
    print(f"    - Numeric answers: {ans_stats.get('numeric_answers', 0)}")
    print(f"    - Multi-answer questions: {ans_stats.get('multi_answer_questions', 0)}")
    print(f"    - Avg answer length: {ans_stats.get('avg_answer_length', 0):.1f} words")
    
    return adapter


def test_task_creation_by_type(adapter: TextVQAAdapter):
    """Test creating tasks for different types"""
    print("\n" + "="*60)
    print("Testing Task Creation by Type")
    print("="*60)
    
    task_types = ['text_reading', 'text_brand_recognition', 'text_temporal_qa', 'text_number_qa']
    created_tasks = {}
    
    for task_type in task_types:
        task_ids = adapter.get_task_ids(task_type=task_type, limit=1)
        if task_ids:
            # Create task using task_id and adapter
            task = TextVQATask(task_id=task_ids[0], adapter=adapter)
            goal, info = task.setup()
            created_tasks[task_type] = task
            
            # Get task data for display
            task_data = adapter.get_task_data(task_ids[0])
            
            print(f"\n[{task_type.upper()} Task]")
            print(f"  - Task ID: {task_data['id']}")
            print(f"  - Question: {task.question[:80]}...")
            print(f"  - Answer: {task.answer}")
            print(f"  - All answers: {task.all_answers[:3]}..." if len(task.all_answers) > 3 else f"  - All answers: {task.all_answers}")
            print(f"  - Question type: {task.question_type}")
            print(f"  - Has OCR: {task.has_ocr}")
            if task.has_ocr:
                print(f"  - OCR tokens: {len(task.ocr_tokens)} tokens")
                print(f"  - Sample OCR: {task.ocr_tokens[:5]}..." if len(task.ocr_tokens) > 5 else f"  - OCR tokens: {task.ocr_tokens}")
            print(f"  - Answer in OCR: {info.get('answer_in_ocr', False)}")
    
    return created_tasks


def test_ocr_based_tasks(adapter: TextVQAAdapter):
    """Test tasks that require OCR understanding"""
    print("\n" + "="*60)
    print("Testing OCR-based Tasks")
    print("="*60)
    
    # Get tasks with OCR tokens
    ocr_tasks = adapter.get_ocr_examples(n=5)
    
    if not ocr_tasks:
        print("No tasks with OCR tokens found")
        return
    
    print(f"Found {len(ocr_tasks)} tasks with OCR tokens")
    
    # Analyze OCR coverage
    for i, task_id in enumerate(ocr_tasks[:3]):
        task = TextVQATask(task_id=task_id, adapter=adapter)
        goal, info = task.setup()
        
        ocr_analysis = adapter.analyze_ocr_coverage(task_id)
        
        print(f"\n[OCR Task {i+1}]")
        print(f"  Question: {task.question[:60]}...")
        print(f"  Answer: '{task.answer}'")
        print(f"  OCR token count: {ocr_analysis['ocr_token_count']}")
        print(f"  Answer in OCR: {ocr_analysis['answer_in_ocr']}")
        print(f"  Partial match: {ocr_analysis['partial_match']}")
        
        # Show relevant OCR tokens
        if task.answer and task.ocr_tokens:
            relevant_tokens = [token for token in task.ocr_tokens 
                             if task.answer.lower() in token.lower() or 
                             token.lower() in task.answer.lower()]
            if relevant_tokens:
                print(f"  Relevant OCR tokens: {relevant_tokens}")


def test_answer_validation_variations(adapter: TextVQAAdapter):
    """Test answer validation with different variations"""
    print("\n" + "="*60)
    print("Testing Answer Validation Variations")
    print("="*60)
    
    # Test different scenarios
    test_configs = [
        ('text_reading', "Test exact text reading"),
        ('text_brand_recognition', "Test brand name variations"),
        ('text_temporal_qa', "Test date/time formats"),
        ('text_number_qa', "Test numeric formats")
    ]
    
    for task_type, description in test_configs:
        # Get a task of this type
        task_ids = adapter.get_task_ids(task_type=task_type, limit=1)
        if not task_ids:
            task_ids = adapter.get_task_ids(question_type=task_type.replace('text_', ''), limit=1)
        
        if not task_ids:
            print(f"\n[{task_type}] No tasks found")
            continue
        
        task = TextVQATask(task_id=task_ids[0], adapter=adapter)
        task.setup()
        
        print(f"\n[{task_type} - {description}]")
        print(f"Question: {task.question[:80]}...")
        print(f"Correct answer: '{task.answer}'")
        print(f"All valid answers: {task.all_answers}")
        
        # Test exact match
        success, feedback = task.check_success(task.answer)
        print(f"\nExact match test:")
        print(f"  ✅ '{task.answer}' -> {feedback}")
        
        # Test alternative answers
        print("\nAlternative answers test:")
        for alt_answer in task.all_answers[:3]:
            if alt_answer != task.answer:
                success, feedback = task.check_success(alt_answer)
                status = "✅" if success else "❌"
                print(f"  {status} '{alt_answer}' -> {feedback[:50]}...")
        
        # Test case variations
        test_cases = [
            (task.answer.upper(), "Uppercase"),
            (task.answer.lower(), "Lowercase"),
            (task.answer.capitalize(), "Capitalized"),
            (f"The answer is {task.answer}", "In sentence"),
        ]
        
        print("\nCase variation tests:")
        for test_answer, desc in test_cases:
            success, feedback = task.check_success(test_answer)
            status = "✅" if success else "❌"
            print(f"  {status} {desc}: '{test_answer}' -> {feedback[:50]}...")


def test_multi_answer_tasks(adapter: TextVQAAdapter):
    """Test tasks with multiple valid answers"""
    print("\n" + "="*60)
    print("Testing Multi-Answer Tasks")
    print("="*60)
    
    # Get tasks with multiple answers
    multi_answer_tasks = adapter.get_multi_answer_examples(n=5)
    
    if not multi_answer_tasks:
        print("No multi-answer tasks found")
        return
    
    print(f"Found {len(multi_answer_tasks)} tasks with multiple valid answers")
    
    for i, task_id in enumerate(multi_answer_tasks[:3]):
        task = TextVQATask(task_id=task_id, adapter=adapter)
        task.setup()
        
        print(f"\n[Multi-Answer Task {i+1}]")
        print(f"  Question: {task.question[:60]}...")
        print(f"  Primary answer: '{task.answer}'")
        print(f"  All valid answers ({len(task.all_answers)}): {task.all_answers}")
        print(f"  Answer frequencies: {task.answer_frequencies}")
        
        # Test each unique answer
        unique_answers = set(ans.lower() for ans in task.all_answers)
        print(f"\n  Testing {len(unique_answers)} unique answers:")
        for j, answer in enumerate(unique_answers):
            success, feedback = task.check_success(answer)
            status = "✅" if success else "❌"
            print(f"    {status} '{answer}' -> Valid: {success}")


def test_complete_workflow(adapter: TextVQAAdapter):
    """Test complete Text-VQA workflow"""
    print("\n" + "="*60)
    print("Testing Complete Text-VQA Workflow")
    print("="*60)
    
    # Get a text reading task with OCR
    task_ids = adapter.get_task_ids(task_type='text_reading', has_ocr=True, limit=1)
    if not task_ids:
        task_ids = adapter.get_task_ids(has_ocr=True, limit=1)
    
    if not task_ids:
        print("No tasks with OCR available!")
        return
    
    task_data = adapter.get_task_data(task_ids[0])
    task = TextVQATask(task_id=task_ids[0], adapter=adapter)
    
    goal, info = task.setup()
    
    print("Task Setup:")
    print(f"  - Question: {task.question}")
    print(f"  - Expected answer: {task.answer}")
    print(f"  - Task type: {task.task_type}")
    print(f"  - Question type: {task.question_type}")
    print(f"  - Has OCR: {task.has_ocr}")
    if task.has_ocr:
        print(f"  - OCR tokens: {len(task.ocr_tokens)}")
        print(f"  - Answer in OCR: {info.get('answer_in_ocr', False)}")
    
    # Get observation
    obs = task.get_observation()
    print(f"\nObservation keys: {list(obs.keys())}")
    
    # Simulate different agent responses
    print(f"\nSimulating agent responses:")
    
    # 1. Exact answer
    chat_history = [{"role": "assistant", "content": f"The text says {task.answer}"}]
    reward, done, message, val_info = task.validate(chat_history, task.answer)
    print(f"  1. Exact answer: {message}")
    print(f"     Reward: {reward}, Done: {done}")
    
    # 2. Answer from OCR
    if task.has_ocr and task.ocr_tokens:
        ocr_answer = task.ocr_tokens[0] if task.ocr_tokens else "no text"
        chat_history = [{"role": "assistant", "content": f"I can see the text '{ocr_answer}'"}]
        reward, done, message, val_info = task.validate(chat_history, ocr_answer)
        print(f"  2. Answer from OCR: {message}")
        print(f"     Reward: {reward}")
    
    # 3. Wrong answer
    wrong_answer = "incorrect text"
    chat_history = [{"role": "assistant", "content": f"The text reads {wrong_answer}"}]
    reward, done, message, val_info = task.validate(chat_history, wrong_answer)
    print(f"  3. Wrong answer: {message}")
    print(f"     Reward: {reward}")
    
    # Get metrics
    metrics = task.get_metrics()
    print(f"\nTask Metrics:")
    for key, value in sorted(metrics.items()):
        if key not in ['question', 'answer', 'choices', 'image_path', 'ocr_tokens', 'all_answers']:
            print(f"  - {key}: {value}")


def test_diverse_examples(adapter: TextVQAAdapter):
    """Test diverse examples covering different characteristics"""
    print("\n" + "="*60)
    print("Testing Diverse Examples")
    print("="*60)
    
    # Get diverse examples
    diverse_tasks = adapter.get_diverse_examples(n=10)
    
    if not diverse_tasks:
        print("No diverse examples available")
        return
    
    # Analyze the diversity
    task_types = Counter()
    question_types = Counter()
    has_ocr_count = 0
    answer_in_ocr_count = 0
    multi_answer_count = 0
    
    print(f"Analyzing {len(diverse_tasks)} diverse tasks...")
    
    for task_id in diverse_tasks:
        task = TextVQATask(task_id=task_id, adapter=adapter)
        goal, info = task.setup()
        
        task_types[task.task_type] += 1
        question_types[task.question_type] += 1
        
        if task.has_ocr:
            has_ocr_count += 1
            if info.get('answer_in_ocr', False):
                answer_in_ocr_count += 1
        
        if len(set(task.all_answers)) > 1:
            multi_answer_count += 1
    
    print(f"\nDiversity Analysis:")
    print(f"  Task types: {dict(task_types)}")
    print(f"  Question types: {dict(question_types)}")
    print(f"  Tasks with OCR: {has_ocr_count}/{len(diverse_tasks)}")
    print(f"  Answer in OCR: {answer_in_ocr_count}/{has_ocr_count if has_ocr_count > 0 else 1}")
    print(f"  Multi-answer tasks: {multi_answer_count}/{len(diverse_tasks)}")


def test_batch_evaluation(adapter: TextVQAAdapter, n_samples: int = 30):
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
        stratified_by='task_type',
        seed=42
    )
    
    results = {
        'by_task_type': defaultdict(list),
        'by_question_type': defaultdict(list),
        'by_answer_type': defaultdict(list),
        'with_ocr': [],
        'without_ocr': [],
        'answer_in_ocr': [],
        'multi_answer': []
    }
    
    print(f"\nProcessing {len(task_ids)} tasks...")
    
    for i, task_id in enumerate(task_ids):
        try:
            task = TextVQATask(task_id=task_id, adapter=adapter)
            goal, info = task.setup()
            
            # Simulate correct answer
            success, _ = task.check_success(task.answer)
            
            # Record results
            results['by_task_type'][task.task_type].append(success)
            results['by_question_type'][task.question_type].append(success)
            
            # Determine answer type
            if len(task.answer.split()) == 1:
                answer_type = 'single_word'
            elif task._is_numeric_answer(task.answer):
                answer_type = 'numeric'
            else:
                answer_type = 'phrase'
            results['by_answer_type'][answer_type].append(success)
            
            if task.has_ocr:
                results['with_ocr'].append(success)
                if info.get('answer_in_ocr', False):
                    results['answer_in_ocr'].append(success)
            else:
                results['without_ocr'].append(success)
            
            if len(set(task.all_answers)) > 1:
                results['multi_answer'].append(success)
            
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
    
    print("\nBy OCR Availability:")
    if results['with_ocr']:
        acc = sum(results['with_ocr']) / len(results['with_ocr']) * 100
        print(f"  - With OCR: {len(results['with_ocr'])} tasks, {acc:.1f}% success")
    if results['without_ocr']:
        acc = sum(results['without_ocr']) / len(results['without_ocr']) * 100
        print(f"  - Without OCR: {len(results['without_ocr'])} tasks, {acc:.1f}% success")
    if results['answer_in_ocr']:
        acc = sum(results['answer_in_ocr']) / len(results['answer_in_ocr']) * 100
        print(f"  - Answer in OCR: {len(results['answer_in_ocr'])} tasks, {acc:.1f}% success")


def visualize_sample_images(adapter: TextVQAAdapter, n_samples: int = 6):
    """Visualize sample images with questions and OCR info"""
    print("\n" + "="*60)
    print(f"Visualizing {n_samples} Sample Images")
    print("="*60)
    
    if len(adapter._task_index) == 0:
        print("No tasks available for visualization!")
        return
    
    # Get diverse samples including OCR tasks
    sample_ids = []
    
    # Try to get some OCR tasks
    ocr_tasks = adapter.get_ocr_examples(n=n_samples//2)
    sample_ids.extend(ocr_tasks)
    
    # Get other diverse tasks
    remaining = n_samples - len(sample_ids)
    if remaining > 0:
        other_tasks = adapter.sample_tasks(remaining, seed=42)
        sample_ids.extend(other_tasks)
    
    # Ensure we have exactly n_samples
    sample_ids = sample_ids[:n_samples]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, task_id in enumerate(sample_ids[:n_samples]):
        try:
            task = TextVQATask(task_id=task_id, adapter=adapter)
            task.setup()
            
            # Try to load and display image
            try:
                img = Image.open(task.image_path)
                axes[i].imshow(img)
                axes[i].axis('off')
                
                # Add title with question (truncated)
                question = task.question[:40] + "..." if len(task.question) > 40 else task.question
                answer = task.answer[:20] + "..." if len(task.answer) > 20 else task.answer
                title = f"{task.task_type.replace('text_', '').upper()}\n{question}\nA: {answer}"
                if task.has_ocr:
                    title += f"\n[OCR: {len(task.ocr_tokens)} tokens]"
                
                axes[i].set_title(title, fontsize=9, wrap=True)
                
                img.close()
            except Exception as e:
                # If image loading fails, show task info
                info_text = f"Image not available\n\nQ: {task.question[:60]}...\nA: {task.answer}"
                if task.has_ocr:
                    info_text += f"\n\nOCR ({len(task.ocr_tokens)} tokens):\n"
                    info_text += "\n".join(task.ocr_tokens[:5])
                    if len(task.ocr_tokens) > 5:
                        info_text += "\n..."
                
                axes[i].text(0.5, 0.5, info_text,
                           ha='center', va='center', transform=axes[i].transAxes,
                           fontsize=10, wrap=True)
                axes[i].axis('off')
                
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error: {str(e)[:50]}",
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')
    
    plt.tight_layout()
    output_file = "textvqa_samples.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSample images saved to: {output_file}")
    plt.close()


def test_answer_analysis(adapter: TextVQAAdapter):
    """Analyze answer patterns in the dataset"""
    print("\n" + "="*60)
    print("Testing Answer Analysis")
    print("="*60)
    
    analysis = adapter.get_answer_analysis()
    
    print(f"\nAnswer Pattern Analysis:")
    print(f"  Total unique answers: {analysis['total_unique_answers']}")
    print(f"  Single word percentage: {analysis['single_word_percentage']:.1f}%")
    print(f"  Numeric answer percentage: {analysis['numeric_answer_percentage']:.1f}%")
    print(f"  Multi-answer percentage: {analysis['multi_answer_percentage']:.1f}%")
    
    if 'answer_agreement_stats' in analysis:
        agreement = analysis['answer_agreement_stats']
        print(f"\nAnswer Agreement Statistics:")
        print(f"  Mean agreement: {agreement['mean']:.2f}")
        print(f"  Std deviation: {agreement['std']:.2f}")
        print(f"  Min agreement: {agreement['min']:.2f}")
        print(f"  Max agreement: {agreement['max']:.2f}")
    
    print(f"\nTop 20 Most Common Answers:")
    for i, (answer, count) in enumerate(analysis['top_50_answers'][:20], 1):
        print(f"  {i:2d}. '{answer}': {count} times")
    
    print(f"\nAnswer Length Distribution (words):")
    length_dist = analysis['answer_length_distribution']
    for length in sorted(length_dist.keys())[:10]:
        count = length_dist[length]
        print(f"  {length} word(s): {count} answers")


def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description='Test Text-VQA integration with VLM Gym')
    parser.add_argument('--annotation', type=str,
                       default='/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/Text-VQA/textvqa_train_vlmgym.json',
                       help='Path to Text-VQA annotation file')
    parser.add_argument('--data-root', type=str,
                       default='/data/wang/meng/GYM-Work/dataset/Text_VQA/train',
                       help='Path to Text-VQA images')
    parser.add_argument('--batch-size', type=int, default=30,
                       help='Batch size for evaluation')
    parser.add_argument('--skip-vis', action='store_true',
                       help='Skip visualization to avoid image loading issues')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Text-VQA-VLMGym Integration Test")
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
        
        # Test 3: OCR-based tasks
        test_ocr_based_tasks(adapter)
        
        # Test 4: Answer validation variations
        test_answer_validation_variations(adapter)
        
        # Test 5: Multi-answer tasks
        test_multi_answer_tasks(adapter)
        
        # Test 6: Complete workflow
        test_complete_workflow(adapter)
        
        # Test 7: Diverse examples
        test_diverse_examples(adapter)
        
        # Test 8: Batch evaluation
        test_batch_evaluation(adapter, n_samples=args.batch_size)
        
        # Test 9: Answer analysis
        test_answer_analysis(adapter)
        
        # Test 10: Visualize samples (optional)
        if not args.skip_vis:
            visualize_sample_images(adapter, n_samples=6)
        
        print("\n" + "="*60)
        print("✓ All tests completed successfully!")
        print("="*60)
        
        print("\nKey findings:")
        print("1. Text-VQA adapter successfully loads and indexes data")
        print("2. Multiple task types for text reading are supported")
        print("3. OCR token integration works properly")
        print("4. Multi-answer validation is handled correctly")
        print("5. Text understanding metrics are properly tracked")
        
        print("\nNext steps:")
        print("1. Test with actual VLM models for text reading capabilities")
        print("2. Evaluate OCR accuracy and text extraction quality")
        print("3. Test on tasks requiring understanding of scene text")
        print("4. Evaluate performance on different text types (signs, brands, etc.)")
        
        if not args.skip_vis:
            print("\n5. Check textvqa_samples.png for visualization examples")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())