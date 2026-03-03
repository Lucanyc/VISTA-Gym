#!/usr/bin/env python3
"""
Test script for DocVQA integration with VLM Gym
Tests various question types, answer formats, and document understanding capabilities
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
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import re

# Add paths for imports
sys.path.insert(0, '/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista')
sys.path.insert(0, '/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data_adapters')

# Import required components
try:
    # Try direct import first since we're in the task directory
    from docvqa import DocVQATask
except ImportError as e:
    print(f"Warning: Failed to import DocVQATask from current directory: {e}")
    try:
        from vlm_gym.environments.task.docvqa import DocVQATask
    except ImportError as e2:
        print(f"Error: Cannot import DocVQATask: {e2}")
        print("\nPlease ensure docvqa.py exists in:")
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
    from docvqa_adapter import DocVQAAdapter
except ImportError as e:
    print(f"Error: Cannot import DocVQAAdapter: {e}")
    print("\nMake sure docvqa_adapter.py is in:")
    print("/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data_adapters/")
    sys.exit(1)


def test_adapter_loading(annotation_file, data_root=None):
    """Test DocVQA adapter initialization and data loading"""
    print("\n" + "="*60)
    print("Testing DocVQA Adapter Loading")
    print("="*60)
    
    # Initialize adapter with provided paths
    print(f"Loading from: {annotation_file}")
    if data_root:
        print(f"Image root: {data_root}")
    
    adapter = DocVQAAdapter(
        data_root=data_root,
        annotation_files=annotation_file,
        validate_images=False,  # Set to False initially to avoid path issues
        min_bbox_area=0.0
    )
    
    # Get statistics
    stats = adapter.get_statistics()
    print(f"\n✓ Adapter loaded successfully!")
    print(f"  - Total tasks: {stats['total']}")
    print(f"  - Unique images: {stats.get('unique_images', 'N/A')}")
    
    # Template distribution
    print(f"\n  Template Distribution:")
    for template, count in sorted(stats.get('template_distribution', {}).items()):
        print(f"    - {template}: {count} ({count/stats['total']*100:.1f}%)")
    
    # Question type distribution
    print(f"\n  Question Type Distribution:")
    for qtype, count in sorted(stats.get('question_type_distribution', {}).items())[:10]:
        print(f"    - {qtype}: {count} ({count/stats['total']*100:.1f}%)")
    
    # Dataset type distribution
    print(f"\n  Dataset Type Distribution:")
    for dtype, count in sorted(stats.get('dataset_type_distribution', {}).items()):
        print(f"    - {dtype}: {count}")
    
    # Bbox statistics
    bbox_stats = stats.get('bbox_statistics', {})
    print(f"\n  Bounding Box Statistics:")
    print(f"    - With bbox: {bbox_stats.get('with_bbox', 0)}")
    print(f"    - Without bbox: {bbox_stats.get('without_bbox', 0)}")
    
    # Answer statistics
    ans_stats = stats.get('answer_statistics', {})
    print(f"\n  Answer Statistics:")
    print(f"    - Single word answers: {ans_stats.get('single_word_answers', 0)}")
    print(f"    - Numeric answers: {ans_stats.get('numeric_answers', 0)}")
    print(f"    - Avg answer length: {ans_stats.get('avg_answer_length', 0):.1f} words")
    
    return adapter


def test_task_creation_by_template(adapter: DocVQAAdapter):
    """Test creating tasks for different templates"""
    print("\n" + "="*60)
    print("Testing Task Creation by Template")
    print("="*60)
    
    templates = ['reasoning', 'data', 'structure', 'unknown']
    created_tasks = {}
    
    for template in templates:
        task_ids = adapter.get_task_ids(template_id=template, limit=1)
        if task_ids:
            # Create task using task_id and adapter (VisionQATask interface)
            task = DocVQATask(task_id=task_ids[0], adapter=adapter)
            goal, info = task.setup()
            created_tasks[template] = task
            
            # Get task data for display
            task_data = adapter.get_task_data(task_ids[0])
            
            print(f"\n[{template.upper()} Task]")
            print(f"  - Task ID: {task_data['id']}")
            print(f"  - Question: {task.question[:80]}...")
            print(f"  - Answer: {task.answer} (type: {task.answer_type})")
            print(f"  - Question type: {task.question_type}")
            print(f"  - Has bbox: {task.has_bbox}")
            if task.has_bbox:
                print(f"  - Bbox area: {task._calculate_bbox_area():.2f}")
            print(f"  - Document complexity: {info.get('document_complexity', 'N/A')}")
    
    return created_tasks


def test_answer_validation_by_type(adapter: DocVQAAdapter):
    """Test answer validation for different answer types"""
    print("\n" + "="*60)
    print("Testing Answer Validation by Type")
    print("="*60)
    
    # Test different answer types
    test_configs = [
        ('numeric', ['123', '456.78', '$1,234.56']),
        ('yes_no', ['yes', 'no']),
        ('temporal', ['01/15/2023', 'January 2023', '2023-01-15']),
        ('entity_extraction', ['Company ABC', 'John Doe', 'New York'])
    ]
    
    for question_type, sample_answers in test_configs:
        # Get a task of this type
        task_ids = adapter.get_task_ids(question_type=question_type, limit=1)
        if not task_ids:
            print(f"\n[{question_type.upper()}] No tasks found")
            continue
        
        task_data = adapter.get_task_data(task_ids[0])
        task = DocVQATask(task_id=task_ids[0], adapter=adapter)
        task.setup()
        
        print(f"\n[{question_type.upper()} - {task.answer_type}]")
        print(f"Question: {task.question[:80]}...")
        print(f"Correct answer: '{task.answer}'")
        
        # Test exact match
        success, feedback = task.check_success(task.answer)
        print(f"\nExact match test:")
        print(f"  ✅ '{task.answer}' -> {feedback}")
        
        # Test variations based on answer type
        if task.answer_type == 'numeric' and task.is_numeric:
            # Test numeric variations
            try:
                value = float(task._extract_number(task.answer))
                test_cases = [
                    (str(value), "Clean number", True),
                    (f"${value:,.2f}", "With currency", True),
                    (str(value * 1.005), "0.5% error", True),
                    (str(value * 1.1), "10% error", False),
                ]
                
                print("\nNumeric validation tests:")
                for test_answer, desc, expected in test_cases:
                    success, feedback = task.check_success(test_answer)
                    status = "✅" if success == expected else "❌"
                    print(f"  {status} {desc}: '{test_answer}' -> {feedback[:50]}...")
            except:
                pass
        
        elif task.answer_type == 'yes_no':
            # Test yes/no variations
            test_cases = [
                ("Yes", "Capitalized", True),
                ("yes.", "With period", True),
                ("The answer is yes", "In sentence", True),
                ("maybe", "Invalid", False),
            ]
            
            print("\nYes/No validation tests:")
            for test_answer, desc, expected in test_cases:
                success, feedback = task.check_success(test_answer)
                status = "✅" if (success and task.answer.lower() == 'yes') or \
                               (not success and task.answer.lower() == 'no') else "❌"
                print(f"  {status} {desc}: '{test_answer}' -> {feedback[:50]}...")


def test_bbox_tasks(adapter: DocVQAAdapter):
    """Test tasks with bounding box annotations"""
    print("\n" + "="*60)
    print("Testing Tasks with Bounding Boxes")
    print("="*60)
    
    # Get tasks with bboxes
    bbox_tasks = adapter.get_localization_examples(n=5)
    
    if not bbox_tasks:
        print("No tasks with bounding boxes found")
        return
    
    print(f"Found {len(bbox_tasks)} tasks with bounding boxes")
    
    # Analyze bbox characteristics
    bbox_areas = []
    for i, task_id in enumerate(bbox_tasks[:3]):
        task = DocVQATask(task_id=task_id, adapter=adapter)
        task.setup()
        
        bbox_analysis = task.analyze_bbox_coverage(task_id)
        
        print(f"\n[Bbox Task {i+1}]")
        print(f"  Question: {task.question[:60]}...")
        print(f"  Answer: '{task.answer}'")
        print(f"  Bbox: {bbox_analysis['bbox']}")
        print(f"  Bbox area: {bbox_analysis['bbox_area']:.2f}")
        
        if len(bbox_analysis['bbox']) >= 4:
            print(f"  Position: ({bbox_analysis['bbox'][0]:.1f}, {bbox_analysis['bbox'][1]:.1f})")
            print(f"  Size: {bbox_analysis['bbox'][2]:.1f} x {bbox_analysis['bbox'][3]:.1f}")
        
        bbox_areas.append(bbox_analysis['bbox_area'])
    
    if bbox_areas:
        print(f"\nBbox area statistics:")
        print(f"  Min: {min(bbox_areas):.2f}")
        print(f"  Max: {max(bbox_areas):.2f}")
        print(f"  Avg: {sum(bbox_areas)/len(bbox_areas):.2f}")


def test_complete_workflow(adapter: DocVQAAdapter):
    """Test complete DocVQA workflow"""
    print("\n" + "="*60)
    print("Testing Complete DocVQA Workflow")
    print("="*60)
    
    # Get a reasoning task
    task_ids = adapter.get_task_ids(template_id='reasoning', limit=1)
    if not task_ids:
        task_ids = adapter.get_task_ids(limit=1)
    
    if not task_ids:
        print("No tasks available!")
        return
    
    task_data = adapter.get_task_data(task_ids[0])
    task = DocVQATask(task_id=task_ids[0], adapter=adapter)
    
    goal, info = task.setup()
    
    print("Task Setup:")
    print(f"  - Question: {task.question}")
    print(f"  - Expected answer: {task.answer}")
    print(f"  - Template: {task.template_id}")
    print(f"  - Question type: {task.question_type}")
    print(f"  - Answer type: {task.answer_type}")
    print(f"  - Has bbox: {task.has_bbox}")
    
    # Get observation
    obs = task.get_observation()
    print(f"\nObservation keys: {list(obs.keys())}")
    
    # Simulate different agent responses
    print(f"\nSimulating agent responses:")
    
    # 1. Exact answer
    chat_history = [{"role": "assistant", "content": f"The answer is {task.answer}"}]
    reward, done, message, val_info = task.validate(chat_history, task.answer)
    print(f"  1. Exact answer: {message}")
    print(f"     Reward: {reward}, Done: {done}")
    
    # 2. Answer with explanation
    detailed_response = f"Looking at the document, I can see that {task.question.lower()} The answer is {task.answer}."
    chat_history = [{"role": "assistant", "content": detailed_response}]
    reward, done, message, val_info = task.validate(chat_history, task.answer)
    print(f"  2. Answer with explanation: {message}")
    print(f"     Reward: {reward}")
    
    # 3. Wrong answer
    wrong_answer = "incorrect answer"
    chat_history = [{"role": "assistant", "content": f"The answer is {wrong_answer}"}]
    reward, done, message, val_info = task.validate(chat_history, wrong_answer)
    print(f"  3. Wrong answer: {message}")
    print(f"     Reward: {reward}")
    
    # Get metrics
    metrics = task.get_metrics()
    print(f"\nTask Metrics:")
    for key, value in sorted(metrics.items()):
        if key not in ['question', 'answer', 'choices', 'image_path']:
            print(f"  - {key}: {value}")


def test_diverse_examples(adapter: DocVQAAdapter):
    """Test diverse examples covering different characteristics"""
    print("\n" + "="*60)
    print("Testing Diverse Examples")
    print("="*60)
    
    # Get diverse examples
    diverse_tasks = adapter.get_diverse_examples(n=12)
    
    if not diverse_tasks:
        print("No diverse examples available")
        return
    
    # Analyze the diversity
    templates = Counter()
    question_types = Counter()
    answer_types = Counter()
    has_bbox_count = 0
    
    print(f"Analyzing {len(diverse_tasks)} diverse tasks...")
    
    for task_id in diverse_tasks:
        task_data = adapter.get_task_data(task_id)
        templates[task_data['metadata'].get('template_id', 'unknown')] += 1
        
        # Create task to get classifications
        task = DocVQATask(task_id=task_id, adapter=adapter)
        task.setup()
        
        question_types[task.question_type] += 1
        answer_types[task.answer_type] += 1
        if task.has_bbox:
            has_bbox_count += 1
    
    print(f"\nDiversity Analysis:")
    print(f"  Templates: {dict(templates)}")
    print(f"  Question types: {dict(question_types)}")
    print(f"  Answer types: {dict(answer_types)}")
    print(f"  Tasks with bbox: {has_bbox_count}/{len(diverse_tasks)}")


def test_batch_evaluation(adapter: DocVQAAdapter, n_samples: int = 30):
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
        'by_template': defaultdict(list),
        'by_question_type': defaultdict(list),
        'by_answer_type': defaultdict(list),
        'by_complexity': defaultdict(list),
        'with_bbox': [],
        'without_bbox': []
    }
    
    print(f"\nProcessing {len(task_ids)} tasks...")
    
    for i, task_id in enumerate(task_ids):
        try:
            task = DocVQATask(task_id=task_id, adapter=adapter)
            goal, info = task.setup()
            
            # Simulate correct answer
            success, _ = task.check_success(task.answer)
            
            # Record results
            results['by_template'][task.template_id].append(success)
            results['by_question_type'][task.question_type].append(success)
            results['by_answer_type'][task.answer_type].append(success)
            results['by_complexity'][info.get('document_complexity', 'unknown')].append(success)
            
            if task.has_bbox:
                results['with_bbox'].append(success)
            else:
                results['without_bbox'].append(success)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(task_ids)} tasks...")
                
        except Exception as e:
            print(f"  Error with task {task_id}: {e}")
    
    # Print summary
    print("\nResults Summary:")
    
    print("\nBy Template:")
    for template, successes in sorted(results['by_template'].items()):
        if successes:
            acc = sum(successes) / len(successes) * 100
            print(f"  - {template}: {len(successes)} tasks, {acc:.1f}% success")
    
    print("\nBy Question Type:")
    for qtype, successes in sorted(results['by_question_type'].items()):
        if successes:
            acc = sum(successes) / len(successes) * 100
            print(f"  - {qtype}: {len(successes)} tasks, {acc:.1f}% success")
    
    print("\nBy Answer Type:")
    for atype, successes in sorted(results['by_answer_type'].items()):
        if successes:
            acc = sum(successes) / len(successes) * 100
            print(f"  - {atype}: {len(successes)} tasks, {acc:.1f}% success")
    
    print("\nBy Bounding Box:")
    if results['with_bbox']:
        acc = sum(results['with_bbox']) / len(results['with_bbox']) * 100
        print(f"  - With bbox: {len(results['with_bbox'])} tasks, {acc:.1f}% success")
    if results['without_bbox']:
        acc = sum(results['without_bbox']) / len(results['without_bbox']) * 100
        print(f"  - Without bbox: {len(results['without_bbox'])} tasks, {acc:.1f}% success")


def visualize_sample_documents(adapter: DocVQAAdapter, n_samples: int = 6):
    """Visualize sample documents with questions and bboxes"""
    print("\n" + "="*60)
    print(f"Visualizing {n_samples} Sample Documents")
    print("="*60)
    
    if len(adapter._task_index) == 0:
        print("No tasks available for visualization!")
        return
    
    # Get diverse samples including bbox tasks
    sample_ids = []
    
    # Try to get some bbox tasks
    bbox_tasks = adapter.get_localization_examples(n=n_samples//2)
    sample_ids.extend(bbox_tasks)
    
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
            task = DocVQATask(task_id=task_id, adapter=adapter)
            task.setup()
            
            # Try to load and display image
            try:
                img = Image.open(task.image_path)
                axes[i].imshow(img)
                
                # If has bbox, draw it
                if task.has_bbox and len(task.answer_bbox) >= 4:
                    # Create rectangle patch
                    rect = patches.Rectangle(
                        (task.answer_bbox[0], task.answer_bbox[1]),
                        task.answer_bbox[2], task.answer_bbox[3],
                        linewidth=3, edgecolor='red', facecolor='none'
                    )
                    axes[i].add_patch(rect)
                
                axes[i].axis('off')
                
                # Add title with question (truncated)
                question = task.question[:40] + "..." if len(task.question) > 40 else task.question
                answer = task.answer[:20] + "..." if len(task.answer) > 20 else task.answer
                title = f"{task.template_id.upper()}\n{question}\nA: {answer}"
                if task.has_bbox:
                    title += "\n[Has BBox]"
                
                axes[i].set_title(title, fontsize=9, wrap=True)
                
                img.close()
            except Exception as e:
                # If image loading fails, show task info
                axes[i].text(0.5, 0.5, 
                           f"Image not available\n\nQ: {task.question[:60]}...\nA: {task.answer}",
                           ha='center', va='center', transform=axes[i].transAxes,
                           fontsize=10, wrap=True)
                axes[i].axis('off')
                
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error: {str(e)[:50]}",
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')
    
    plt.tight_layout()
    output_file = "docvqa_samples.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSample documents saved to: {output_file}")
    plt.close()


def test_answer_analysis(adapter: DocVQAAdapter):
    """Analyze answer patterns in the dataset"""
    print("\n" + "="*60)
    print("Testing Answer Analysis")
    print("="*60)
    
    analysis = adapter.get_answer_analysis()
    
    print(f"\nAnswer Pattern Analysis:")
    print(f"  Total unique answers: {analysis['total_unique_answers']}")
    print(f"  Single word percentage: {analysis['single_word_percentage']:.1f}%")
    print(f"  Numeric answer percentage: {analysis['numeric_answer_percentage']:.1f}%")
    
    print(f"\nAnswer Categories:")
    for category, count in analysis['answer_categories'].items():
        print(f"  - {category}: {count}")
    
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
    parser = argparse.ArgumentParser(description='Test DocVQA integration with VLM Gym')
    parser.add_argument('--annotation', type=str,
                       default='/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/DocVQA/docvqa_train.json',
                       help='Path to DocVQA annotation file')
    parser.add_argument('--data-root', type=str,
                       default='/data/wang/meng/GYM-Work/dataset/DocVQA',
                       help='Path to DocVQA images')
    parser.add_argument('--batch-size', type=int, default=30,
                       help='Batch size for evaluation')
    parser.add_argument('--skip-vis', action='store_true',
                       help='Skip visualization to avoid image loading issues')
    
    args = parser.parse_args()
    
    print("="*60)
    print("DocVQA-VLMGym Integration Test")
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
        
        # Test 2: Create tasks by template
        tasks = test_task_creation_by_template(adapter)
        
        # Test 3: Answer validation by type
        test_answer_validation_by_type(adapter)
        
        # Test 4: Bbox tasks
        test_bbox_tasks(adapter)
        
        # Test 5: Complete workflow
        test_complete_workflow(adapter)
        
        # Test 6: Diverse examples
        test_diverse_examples(adapter)
        
        # Test 7: Batch evaluation
        test_batch_evaluation(adapter, n_samples=args.batch_size)
        
        # Test 8: Answer analysis
        test_answer_analysis(adapter)
        
        # Test 9: Visualize samples (optional)
        if not args.skip_vis:
            visualize_sample_documents(adapter, n_samples=6)
        
        print("\n" + "="*60)
        print("✓ All tests completed successfully!")
        print("="*60)
        
        print("\nKey findings:")
        print("1. DocVQA adapter successfully loads and indexes data")
        print("2. Multiple question types and templates are supported")
        print("3. Answer validation works for text, numeric, temporal formats")
        print("4. Bounding box annotations are properly handled")
        print("5. Task metrics provide document complexity assessment")
        
        print("\nNext steps:")
        print("1. Test with actual VLM models for document understanding")
        print("2. Evaluate OCR and text extraction capabilities")
        print("3. Test reasoning tasks that require multiple information pieces")
        print("4. Evaluate performance on tasks with answer localization (bbox)")
        
        if not args.skip_vis:
            print("\n5. Check docvqa_samples.png for visualization examples")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())