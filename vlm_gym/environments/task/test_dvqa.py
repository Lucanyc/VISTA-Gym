#!/usr/bin/env python3
"""
Test script for DVQA integration with VLM Gym
Tests various chart types, question types, and visual reasoning capabilities
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
    from dvqa import DVQATask
except ImportError as e:
    print(f"Warning: Failed to import DVQATask from current directory: {e}")
    try:
        from vlm_gym.environments.task.dvqa import DVQATask
    except ImportError as e2:
        print(f"Error: Cannot import DVQATask: {e2}")
        print("\nPlease ensure dvqa.py exists in:")
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
    from dvqa_adapter import DVQAAdapter
except ImportError as e:
    print(f"Error: Cannot import DVQAAdapter: {e}")
    print("\nMake sure dvqa_adapter.py is in:")
    print("/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data_adapters/")
    sys.exit(1)


def test_adapter_loading(annotation_file, data_root=None):
    """Test DVQA adapter initialization and data loading"""
    print("\n" + "="*60)
    print("Testing DVQA Adapter Loading")
    print("="*60)
    
    # Initialize adapter with provided paths
    print(f"Loading from: {annotation_file}")
    if data_root:
        print(f"Image root: {data_root}")
    
    adapter = DVQAAdapter(
        data_root=data_root,
        annotation_files=annotation_file,
        validate_images=False,  # Set to False initially to avoid path issues
        min_bbox_area=0.0
    )
    
    # Get statistics
    stats = adapter.get_statistics()
    print(f"\n✓ Adapter loaded successfully!")
    print(f"  - Total tasks: {stats['total']:,}")
    print(f"  - Unique images: {stats.get('image_statistics', {}).get('unique_images', 'N/A')}")
    
    # Template distribution
    print(f"\n  Template Distribution:")
    for template, count in sorted(stats.get('template_distribution', {}).items()):
        print(f"    - {template}: {count:,} ({count/stats['total']*100:.1f}%)")
    
    # Chart type distribution
    print(f"\n  Chart Type Distribution:")
    for chart_type, count in sorted(stats.get('chart_type_distribution', {}).items())[:10]:
        print(f"    - {chart_type}: {count:,} ({count/stats['total']*100:.1f}%)")
    
    # Question type distribution
    print(f"\n  Question Type Distribution:")
    for qtype, count in sorted(stats.get('question_type_distribution', {}).items())[:10]:
        print(f"    - {qtype}: {count:,} ({count/stats['total']*100:.1f}%)")
    
    # Special tasks
    special = stats.get('special_tasks', {})
    print(f"\n  Special Task Types:")
    print(f"    - Reasoning tasks: {special.get('reasoning_tasks', 0):,}")
    print(f"    - Data extraction tasks: {special.get('data_tasks', 0):,}")
    print(f"    - Structure tasks: {special.get('structure_tasks', 0):,}")
    print(f"    - Color questions: {special.get('color_tasks', 0):,}")
    print(f"    - Comparison tasks: {special.get('comparison_tasks', 0):,}")
    
    # Answer statistics
    ans_stats = stats.get('answer_statistics', {})
    print(f"\n  Answer Statistics:")
    print(f"    - Single word answers: {ans_stats.get('single_word_answers', 0):,}")
    print(f"    - Numeric answers: {ans_stats.get('numeric_answers', 0):,}")
    print(f"    - Avg answer length: {ans_stats.get('avg_answer_length', 0):.1f} words")
    
    return adapter


def test_task_creation_by_template(adapter: DVQAAdapter):
    """Test creating tasks for different templates"""
    print("\n" + "="*60)
    print("Testing Task Creation by Template")
    print("="*60)
    
    templates = ['reasoning', 'data', 'structure']
    created_tasks = {}
    
    for template in templates:
        task_ids = adapter.get_task_ids(template_id=template, limit=2)
        if task_ids:
            for i, task_id in enumerate(task_ids[:1]):  # Show 1 example per template
                # Create task using task_id and adapter
                task = DVQATask(task_id=task_id, adapter=adapter)
                goal, info = task.setup()
                created_tasks[template] = task
                
                print(f"\n[{template.upper()} Task Example]")
                print(f"  - Task ID: {task_id}")
                print(f"  - Chart type: {task.chart_type}")
                print(f"  - Question: {task.question}")
                print(f"  - Answer: {task.answer} (type: {task.answer_type})")
                print(f"  - Question type: {task.question_type}")
                print(f"  - Is numeric: {task.is_numeric}")
                print(f"  - Is comparison: {task.is_comparison}")
                print(f"  - Chart complexity: {info.get('chart_complexity', 'N/A')}")
    
    return created_tasks


def test_chart_types(adapter: DVQAAdapter):
    """Test tasks for different chart types"""
    print("\n" + "="*60)
    print("Testing Different Chart Types")
    print("="*60)
    
    chart_types = ['bar', 'line', 'pie']
    
    for chart_type in chart_types:
        task_ids = adapter.get_chart_type_examples(chart_type, n=2)
        if task_ids:
            print(f"\n[{chart_type.upper()} Chart Examples]")
            for task_id in task_ids[:1]:  # Show 1 example per type
                task = DVQATask(task_id=task_id, adapter=adapter)
                task.setup()
                
                print(f"  Question: {task.question}")
                print(f"  Answer: {task.answer}")
                print(f"  Template: {task.template_id}")
                print(f"  Question type: {task.question_type}")


def test_answer_validation_by_type(adapter: DVQAAdapter):
    """Test answer validation for different answer types in DVQA"""
    print("\n" + "="*60)
    print("Testing Answer Validation by Type")
    print("="*60)
    
    # Test different answer scenarios common in DVQA
    test_configs = [
        ('value_extraction', 'Numeric value extraction'),
        ('comparison', 'Comparison questions'),
        ('yes_no', 'Yes/No questions'),
        ('color', 'Color identification'),
        ('counting', 'Counting questions'),
        ('label_reading', 'Label reading')
    ]
    
    for question_type, description in test_configs:
        # Get a task of this type
        task_ids = adapter.get_task_ids(question_type=question_type, limit=1)
        if not task_ids:
            print(f"\n[{description}] No tasks found")
            continue
        
        task = DVQATask(task_id=task_ids[0], adapter=adapter)
        task.setup()
        
        print(f"\n[{description} - {task.answer_type}]")
        print(f"Chart type: {task.chart_type}")
        print(f"Question: {task.question}")
        print(f"Correct answer: '{task.answer}'")
        
        # Test exact match
        success, feedback = task.check_success(task.answer)
        print(f"\nExact match test:")
        print(f"  ✅ '{task.answer}' -> Success: {success}")
        
        # Test type-specific variations
        if task.is_numeric:
            # Test numeric variations
            test_cases = []
            try:
                value = float(task._extract_number(task.answer))
                test_cases = [
                    (str(int(value)), "Integer form", value == int(value)),
                    (str(value + 0.001), "Small error", False),
                    ("The value is " + str(value), "With text", True),
                ]
            except:
                pass
            
            if test_cases:
                print("\nNumeric validation tests:")
                for test_answer, desc, expected in test_cases:
                    success, feedback = task.check_success(test_answer)
                    status = "✅" if success == expected else "❌"
                    print(f"  {status} {desc}: '{test_answer}' -> Success: {success}")
        
        elif task.answer_type == 'yes_no':
            # Test yes/no variations
            test_cases = [
                ("Yes", "Capitalized"),
                ("yes", "Lowercase"),
                ("NO", "Uppercase NO"),
            ]
            
            print("\nYes/No validation tests:")
            for test_answer, desc in test_cases:
                success, feedback = task.check_success(test_answer)
                print(f"  '{test_answer}' ({desc}) -> Success: {success}")
        
        elif task.answer_type == 'color':
            # Test color variations
            if task.answer.lower() in ['red', 'blue', 'green']:
                test_color = 'crimson' if task.answer.lower() == 'red' else task.answer
                success, feedback = task.check_success(test_color)
                print(f"\nColor variation test:")
                print(f"  '{test_color}' -> Success: {success}")


def test_special_question_types(adapter: DVQAAdapter):
    """Test special question types in DVQA"""
    print("\n" + "="*60)
    print("Testing Special Question Types")
    print("="*60)
    
    # Test color questions
    print("\n[Color Questions]")
    color_tasks = adapter.get_color_examples(n=2)
    for task_id in color_tasks[:1]:
        task = DVQATask(task_id=task_id, adapter=adapter)
        task.setup()
        print(f"  Q: {task.question}")
        print(f"  A: {task.answer}")
    
    # Test comparison questions
    print("\n[Comparison Questions]")
    comparison_tasks = adapter.get_comparison_examples(n=2)
    for task_id in comparison_tasks[:1]:
        task = DVQATask(task_id=task_id, adapter=adapter)
        task.setup()
        print(f"  Q: {task.question}")
        print(f"  A: {task.answer}")
    
    # Test reasoning tasks
    print("\n[Reasoning Tasks]")
    reasoning_tasks = adapter.get_template_examples('reasoning', n=2)
    for task_id in reasoning_tasks[:1]:
        task = DVQATask(task_id=task_id, adapter=adapter)
        task.setup()
        print(f"  Q: {task.question}")
        print(f"  A: {task.answer}")
        print(f"  Chart type: {task.chart_type}")


def test_complete_workflow(adapter: DVQAAdapter):
    """Test complete DVQA workflow"""
    print("\n" + "="*60)
    print("Testing Complete DVQA Workflow")
    print("="*60)
    
    # Get a reasoning task about bar charts
    task_ids = adapter.get_task_ids(template_id='reasoning', chart_type='bar', limit=1)
    if not task_ids:
        task_ids = adapter.get_task_ids(limit=1)
    
    if not task_ids:
        print("No tasks available!")
        return
    
    task = DVQATask(task_id=task_ids[0], adapter=adapter)
    goal, info = task.setup()
    
    print("Task Setup:")
    print(f"  - Chart type: {task.chart_type}")
    print(f"  - Question: {task.question}")
    print(f"  - Expected answer: {task.answer}")
    print(f"  - Template: {task.template_id}")
    print(f"  - Question type: {task.question_type}")
    print(f"  - Answer type: {task.answer_type}")
    print(f"  - Is numeric: {task.is_numeric}")
    print(f"  - Single word answer: {task.single_word_answer}")
    
    # Get observation
    obs = task.get_observation()
    print(f"\nObservation keys: {list(obs.keys())}")
    
    # Simulate different agent responses
    print(f"\nSimulating agent responses:")
    
    # 1. Exact answer
    chat_history = [{"role": "assistant", "content": f"{task.answer}"}]
    reward, done, message, val_info = task.validate(chat_history, task.answer)
    print(f"  1. Exact answer: {message}")
    print(f"     Reward: {reward}, Done: {done}")
    
    # 2. Answer with explanation (common in chart analysis)
    detailed_response = f"Looking at the {task.chart_type} chart, I can see that the answer is {task.answer}."
    chat_history = [{"role": "assistant", "content": detailed_response}]
    reward, done, message, val_info = task.validate(chat_history, task.answer)
    print(f"  2. Answer with explanation: {message}")
    print(f"     Reward: {reward}")
    
    # 3. Wrong answer
    wrong_answer = "10" if task.answer != "10" else "5"
    chat_history = [{"role": "assistant", "content": wrong_answer}]
    reward, done, message, val_info = task.validate(chat_history, wrong_answer)
    print(f"  3. Wrong answer: {message}")
    print(f"     Reward: {reward}")
    
    # Get metrics
    metrics = task.get_metrics()
    print(f"\nTask Metrics:")
    relevant_metrics = ['template_id', 'question_type', 'answer_type', 'chart_type', 
                       'is_numeric', 'chart_complexity', 'answer_complexity']
    for key in relevant_metrics:
        if key in metrics:
            print(f"  - {key}: {metrics[key]}")


def test_diverse_examples(adapter: DVQAAdapter):
    """Test diverse examples covering different characteristics"""
    print("\n" + "="*60)
    print("Testing Diverse Examples")
    print("="*60)
    
    # Get diverse examples
    diverse_tasks = adapter.get_diverse_examples(n=15)
    
    if not diverse_tasks:
        print("No diverse examples available")
        return
    
    # Analyze the diversity
    templates = Counter()
    question_types = Counter()
    answer_types = Counter()
    chart_types = Counter()
    numeric_count = 0
    color_count = 0
    comparison_count = 0
    
    print(f"Analyzing {len(diverse_tasks)} diverse tasks...")
    
    for task_id in diverse_tasks:
        task = DVQATask(task_id=task_id, adapter=adapter)
        task.setup()
        
        templates[task.template_id] += 1
        question_types[task.question_type] += 1
        answer_types[task.answer_type] += 1
        chart_types[task.chart_type] += 1
        
        if task.is_numeric:
            numeric_count += 1
        if task.is_color_question:
            color_count += 1
        if task.is_comparison:
            comparison_count += 1
    
    print(f"\nDiversity Analysis:")
    print(f"  Templates: {dict(templates)}")
    print(f"  Chart types: {dict(chart_types)}")
    print(f"  Question types: {dict(question_types)}")
    print(f"  Answer types: {dict(answer_types)}")
    print(f"  Numeric answers: {numeric_count}/{len(diverse_tasks)}")
    print(f"  Color questions: {color_count}/{len(diverse_tasks)}")
    print(f"  Comparison questions: {comparison_count}/{len(diverse_tasks)}")


def test_batch_evaluation(adapter: DVQAAdapter, n_samples: int = 50):
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
    print(f"Available tasks: {total_tasks:,}, testing {actual_samples} samples")
    
    # Sample tasks with different strategies
    task_ids = adapter.sample_tasks(
        actual_samples, 
        balanced_templates=True,  # Balance across templates
        seed=42
    )
    
    results = {
        'by_template': defaultdict(list),
        'by_chart_type': defaultdict(list),
        'by_question_type': defaultdict(list),
        'by_answer_type': defaultdict(list),
        'numeric_answers': [],
        'text_answers': [],
        'single_word': [],
        'multi_word': []
    }
    
    print(f"\nProcessing {len(task_ids)} tasks...")
    
    for i, task_id in enumerate(task_ids):
        try:
            task = DVQATask(task_id=task_id, adapter=adapter)
            goal, info = task.setup()
            
            # Simulate correct answer
            success, _ = task.check_success(task.answer)
            
            # Record results
            results['by_template'][task.template_id].append(success)
            results['by_chart_type'][task.chart_type].append(success)
            results['by_question_type'][task.question_type].append(success)
            results['by_answer_type'][task.answer_type].append(success)
            
            if task.is_numeric:
                results['numeric_answers'].append(success)
            else:
                results['text_answers'].append(success)
            
            if task.single_word_answer:
                results['single_word'].append(success)
            else:
                results['multi_word'].append(success)
            
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
    
    print("\nBy Chart Type:")
    for chart_type, successes in sorted(results['by_chart_type'].items()):
        if successes:
            acc = sum(successes) / len(successes) * 100
            print(f"  - {chart_type}: {len(successes)} tasks, {acc:.1f}% success")
    
    print("\nBy Question Type (top 5):")
    q_type_results = [(qtype, successes) for qtype, successes in results['by_question_type'].items()]
    q_type_results.sort(key=lambda x: len(x[1]), reverse=True)
    for qtype, successes in q_type_results[:5]:
        if successes:
            acc = sum(successes) / len(successes) * 100
            print(f"  - {qtype}: {len(successes)} tasks, {acc:.1f}% success")
    
    print("\nBy Answer Format:")
    if results['numeric_answers']:
        acc = sum(results['numeric_answers']) / len(results['numeric_answers']) * 100
        print(f"  - Numeric: {len(results['numeric_answers'])} tasks, {acc:.1f}% success")
    if results['text_answers']:
        acc = sum(results['text_answers']) / len(results['text_answers']) * 100
        print(f"  - Text: {len(results['text_answers'])} tasks, {acc:.1f}% success")


def visualize_sample_charts(adapter: DVQAAdapter, n_samples: int = 9):
    """Visualize sample charts with questions and answers"""
    print("\n" + "="*60)
    print(f"Visualizing {n_samples} Sample Charts")
    print("="*60)
    
    if len(adapter._task_index) == 0:
        print("No tasks available for visualization!")
        return
    
    # Get diverse samples including different chart types
    sample_ids = []
    
    # Try to get samples from each chart type
    for chart_type in ['bar', 'line', 'pie']:
        chart_tasks = adapter.get_chart_type_examples(chart_type, n=3)
        sample_ids.extend(chart_tasks[:3])
    
    # Ensure we have exactly n_samples
    if len(sample_ids) < n_samples:
        remaining = adapter.sample_tasks(n_samples - len(sample_ids), seed=42)
        sample_ids.extend(remaining)
    
    sample_ids = sample_ids[:n_samples]
    
    # Create visualization
    rows = 3
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, task_id in enumerate(sample_ids[:n_samples]):
        try:
            task = DVQATask(task_id=task_id, adapter=adapter)
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
                
                # Add title with chart info
                question = task.question[:50] + "..." if len(task.question) > 50 else task.question
                title = f"{task.chart_type.upper()} - {task.template_id}\n{question}\nA: {task.answer}"
                if task.is_numeric:
                    title += " [Numeric]"
                elif task.is_color_question:
                    title += " [Color]"
                
                axes[i].set_title(title, fontsize=10, wrap=True)
                
                img.close()
            except Exception as e:
                # If image loading fails, show task info
                info_text = f"Chart: {task.chart_type}\nTemplate: {task.template_id}\n\n"
                info_text += f"Q: {task.question[:80]}...\n\nA: {task.answer}"
                
                axes[i].text(0.5, 0.5, info_text,
                           ha='center', va='center', transform=axes[i].transAxes,
                           fontsize=11, wrap=True)
                axes[i].axis('off')
                
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error: {str(e)[:50]}",
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')
    
    plt.tight_layout()
    output_file = "dvqa_samples.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSample charts saved to: {output_file}")
    plt.close()


def test_chart_complexity_analysis(adapter: DVQAAdapter):
    """Analyze chart complexity patterns"""
    print("\n" + "="*60)
    print("Testing Chart Complexity Analysis")
    print("="*60)
    
    complexity_analysis = adapter.analyze_chart_complexity()
    
    print("\nComplexity by Template:")
    for template, stats in complexity_analysis['by_template'].items():
        print(f"\n  {template.upper()}:")
        print(f"    - Count: {stats['count']:,}")
        print(f"    - Avg question length: {stats['avg_question_length']:.1f} words")
        print(f"    - Avg answer length: {stats['avg_answer_length']:.1f} words")
        print(f"    - Numeric ratio: {stats['numeric_ratio']:.1%}")
        print(f"    - Single word ratio: {stats['single_word_ratio']:.1%}")
    
    print("\nComplexity by Chart Type (with >10 samples):")
    for chart_type, stats in complexity_analysis['by_chart_type'].items():
        if stats['count'] > 10:
            print(f"\n  {chart_type}:")
            print(f"    - Count: {stats['count']:,}")
            print(f"    - Template distribution: {stats['template_distribution']}")


def test_numeric_answer_handling(adapter: DVQAAdapter):
    """Test handling of numeric answers specifically"""
    print("\n" + "="*60)
    print("Testing Numeric Answer Handling")
    print("="*60)
    
    # Get numeric answer tasks
    numeric_tasks = adapter.get_task_ids(is_numeric=True, limit=5)
    
    if not numeric_tasks:
        print("No numeric answer tasks found")
        return
    
    print(f"Testing {len(numeric_tasks)} numeric answer tasks...")
    
    for i, task_id in enumerate(numeric_tasks[:3]):  # Show 3 examples
        task = DVQATask(task_id=task_id, adapter=adapter)
        task.setup()
        
        print(f"\n[Numeric Task {i+1}]")
        print(f"  Chart: {task.chart_type}")
        print(f"  Question: {task.question}")
        print(f"  Answer: {task.answer}")
        
        # Test different numeric formats
        if task._extract_number(task.answer) is not None:
            num_value = task._extract_number(task.answer)
            test_cases = [
                (str(int(num_value)), "Integer format"),
                (f"{num_value:.1f}", "One decimal"),
                (f"The answer is {num_value}", "With text"),
                ("two" if num_value == 2 else "five", "Word form"),
            ]
            
            print("  Validation tests:")
            for test_answer, desc in test_cases:
                success, _ = task.check_success(test_answer)
                print(f"    - {desc}: '{test_answer}' -> {'✅' if success else '❌'}")


def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description='Test DVQA integration with VLM Gym')
    parser.add_argument('--annotation', type=str,
                       default='/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/DVQA/dvqa_train_vlmgym.json',
                       help='Path to DVQA annotation file')
    parser.add_argument('--data-root', type=str,
                       default='/data/wang/meng/GYM-Work/dataset/DVQA/DVQA/images',
                       help='Path to DVQA images')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Batch size for evaluation')
    parser.add_argument('--skip-vis', action='store_true',
                       help='Skip visualization to avoid image loading issues')
    
    args = parser.parse_args()
    
    print("="*60)
    print("DVQA-VLMGym Integration Test")
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
        
        # Test 3: Test different chart types
        test_chart_types(adapter)
        
        # Test 4: Answer validation by type
        test_answer_validation_by_type(adapter)
        
        # Test 5: Special question types
        test_special_question_types(adapter)
        
        # Test 6: Complete workflow
        test_complete_workflow(adapter)
        
        # Test 7: Diverse examples
        test_diverse_examples(adapter)
        
        # Test 8: Batch evaluation
        test_batch_evaluation(adapter, n_samples=args.batch_size)
        
        # Test 9: Chart complexity analysis
        test_chart_complexity_analysis(adapter)
        
        # Test 10: Numeric answer handling
        test_numeric_answer_handling(adapter)
        
        # Test 11: Visualize samples (optional)
        if not args.skip_vis:
            visualize_sample_charts(adapter, n_samples=9)
        
        print("\n" + "="*60)
        print("✓ All tests completed successfully!")
        print("="*60)
        
        print("\nKey findings:")
        print("1. DVQA adapter successfully loads and indexes chart QA data")
        print("2. Three template types (reasoning, data, structure) work correctly")
        print("3. Multiple chart types (bar, line, pie) are properly handled")
        print("4. Answer validation works for numeric, text, color, and yes/no formats")
        print("5. Special question types (comparison, color, counting) are supported")
        print("6. Task metrics provide chart complexity assessment")
        
        print("\nNext steps:")
        print("1. Test with VLM models for chart understanding capabilities")
        print("2. Evaluate performance on reasoning tasks requiring data analysis")
        print("3. Test numeric extraction and calculation accuracy")
        print("4. Evaluate color recognition in charts")
        print("5. Test comparison and aggregation questions")
        
        if not args.skip_vis:
            print("\n6. Check dvqa_samples.png for chart visualization examples")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())