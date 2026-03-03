#!/usr/bin/env python3
"""
Test script for AI2D integration with VLMGym
Tests science diagram understanding with multiple choice questions
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import random
from collections import defaultdict, Counter
import time
import traceback
from PIL import Image
import matplotlib.pyplot as plt

# Add VLMGym to path
sys.path.insert(0, '/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista')

# Import required components
from vlm_gym.environments.task.ai2d import AI2DTask
from vlm_gym.environments.task.vision_qa_task import VisionQATask
from data_adapters.ai2d_adapter import AI2DAdapter


def test_adapter_loading():
    """Test AI2D adapter initialization and data loading"""
    print("\n" + "="*60)
    print("Testing AI2D Adapter Loading")
    print("="*60)
    
    # Initialize adapter
    adapter = AI2DAdapter(
        data_root="/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/AI2D",
        annotation_files="/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/AI2D/ai2d_train.json",
        validate_images=True
    )
    
    # Get statistics
    stats = adapter.get_statistics()
    print(f"\n✓ Adapter loaded successfully!")
    print(f"  - Total tasks: {stats['total']:,}")
    print(f"  - Unique images: {stats.get('unique_images', 'N/A'):,}")
    print(f"  - Missing images: {stats.get('missing_images', 0)}")
    
    # Split distribution
    print(f"\n  Split Distribution:")
    for split, count in sorted(stats.get('split_distribution', {}).items()):
        percentage = (count / stats['total']) * 100
        print(f"    - {split}: {count:,} ({percentage:.1f}%)")
    
    # Question type distribution
    print(f"\n  Question Type Distribution:")
    for qtype, count in sorted(stats.get('question_type_distribution', {}).items(), 
                              key=lambda x: -x[1])[:5]:
        percentage = (count / stats['total']) * 100
        print(f"    - {qtype}: {count:,} ({percentage:.1f}%)")
    
    # Answer distribution
    print(f"\n  Answer Distribution:")
    for letter, count in sorted(stats.get('answer_distribution', {}).items()):
        if letter != 'unknown':
            percentage = (count / stats['total']) * 100
            print(f"    - {letter}: {count:,} ({percentage:.1f}%)")
    
    # Top keywords
    print(f"\n  Top Science Keywords:")
    for keyword, count in list(stats.get('top_keywords', {}).items())[:10]:
        print(f"    - {keyword}: {count}")
    
    return adapter


def test_task_creation(adapter: AI2DAdapter):
    """Test creating AI2D tasks of different types"""
    print("\n" + "="*60)
    print("Testing Task Creation")
    print("="*60)
    
    # Test different question types
    question_types = ['identification', 'counting', 'comparison', 'spatial', 'process']
    
    created_tasks = []
    
    for qtype in question_types:
        task_ids = adapter.get_task_ids(question_type=qtype, limit=1)
        if not task_ids:
            print(f"\n[{qtype.capitalize()}] - No tasks found")
            continue
            
        task_id = task_ids[0]
        task = AI2DTask(task_id=task_id, adapter=adapter)
        goal, info = task.setup()
        
        print(f"\n[{qtype.capitalize()} Task]")
        print(f"  - Task ID: {task_id}")
        print(f"  - Question: {task.question}")
        print(f"  - Choices: {task.choices}")
        print(f"  - Answer: {task.answer} (Letter: {task.answer_letter})")
        print(f"  - Question Category: {info.get('question_category', 'N/A')}")
        print(f"  - Science Domain: {info.get('science_domain', 'N/A')}")
        print(f"  - Complexity: {info.get('question_complexity', 'N/A')}")
        
        created_tasks.append(task)
    
    return created_tasks[0] if created_tasks else None


def test_filtering_capabilities(adapter: AI2DAdapter):
    """Test various filtering options"""
    print("\n" + "="*60)
    print("Testing Filtering Capabilities")
    print("="*60)
    
    # Test split filtering
    print("\n[Split Filtering]")
    for split in ['train', 'val', 'test']:
        task_ids = adapter.get_task_ids(split=split, limit=10)
        if task_ids:
            print(f"  - {split}: {len(task_ids)} tasks")
    
    # Test question type filtering
    print("\n[Question Type Filtering]")
    for qtype in ['identification', 'counting', 'comparison', 'spatial']:
        task_ids = adapter.get_task_ids(question_type=qtype, limit=10)
        if task_ids:
            print(f"  - {qtype}: {len(task_ids)} tasks")
            # Show a sample
            sample_data = adapter.get_task_data(task_ids[0])
            print(f"    Sample: {sample_data['question'][:60]}...")
    
    # Test answer letter filtering
    print("\n[Answer Letter Filtering]")
    for letter in ['A', 'B', 'C', 'D']:
        task_ids = adapter.get_task_ids(answer_letter=letter, limit=100)
        if task_ids:
            percentage = len(task_ids) / adapter.stats['total'] * 100
            print(f"  - Answer {letter}: {len(task_ids)} tasks ({percentage:.1f}%)")
    
    # Test keyword filtering
    print("\n[Science Domain Filtering by Keyword]")
    keywords = ['cell', 'force', 'earth', 'chemical', 'plant']
    for keyword in keywords:
        task_ids = adapter.get_task_ids(keyword=keyword, limit=10)
        if task_ids:
            print(f"  - {keyword}: {len(task_ids)} tasks")


def test_answer_validation(adapter: AI2DAdapter):
    """Test answer validation with various formats"""
    print("\n" + "="*60)
    print("Testing Answer Validation")
    print("="*60)
    
    # Get a sample task
    task_ids = adapter.get_task_ids(limit=1)
    if not task_ids:
        print("No tasks available for testing")
        return
    
    task = AI2DTask(task_id=task_ids[0], adapter=adapter)
    task.setup()
    
    print(f"Question: {task.question}")
    print(f"Choices: {task.choices}")
    print(f"Correct answer: {task.answer} (Letter: {task.answer_letter})")
    
    # Test cases
    test_cases = [
        # Correct answers
        (task.answer, "Direct content answer ✓"),
        (task.answer_letter, "Letter answer ✓"),
        (task.answer.lower(), "Lowercase content ✓"),
        (f"The answer is {task.answer}", "Content in sentence ✓"),
        (f"I think it's {task.answer_letter}", "Letter in sentence ✓"),
        
        # Wrong answers
        ("A" if task.answer_letter != "A" else "B", "Wrong letter ✗"),
        (task.choices[0] if task.choices[0] != task.answer else task.choices[1], "Wrong choice ✗"),
        ("", "Empty answer ✗"),
        ("ABCD", "Multiple letters ✗"),
        ("random text", "Random text ✗"),
    ]
    
    print("\nTesting various answer formats:")
    print("(✓ = should pass, ✗ = should fail)")
    for answer, description in test_cases:
        success, feedback = task.check_success(answer)
        expected = "✓" in description
        actual = "✓" if success else "✗"
        match = "✅" if (expected == success) else "❌"
        print(f"  {match} {description}: '{answer}' -> {feedback[:50]}...")


def test_science_domain_analysis(adapter: AI2DAdapter):
    """Analyze science domains in the dataset"""
    print("\n" + "="*60)
    print("Testing Science Domain Analysis")
    print("="*60)
    
    # Get samples from different domains
    domains = {
        'biology': adapter.get_biology_tasks(n=50),
        'physics': adapter.get_physics_tasks(n=50),
        'earth_science': adapter.get_earth_science_tasks(n=50)
    }
    
    domain_stats = defaultdict(lambda: {
        'total': 0,
        'question_types': Counter(),
        'avg_choices': 0,
        'answer_distribution': Counter()
    })
    
    for domain_name, task_ids in domains.items():
        if not task_ids:
            continue
            
        print(f"\n[{domain_name.replace('_', ' ').title()}]")
        print(f"  Found {len(task_ids)} tasks")
        
        total_choices = 0
        for task_id in task_ids:
            task = AI2DTask(task_id=task_id, adapter=adapter)
            goal, info = task.setup()
            
            domain_stats[domain_name]['total'] += 1
            domain_stats[domain_name]['question_types'][info.get('question_category', 'unknown')] += 1
            domain_stats[domain_name]['answer_distribution'][task.answer_letter] += 1
            total_choices += len(task.choices)
        
        if domain_stats[domain_name]['total'] > 0:
            domain_stats[domain_name]['avg_choices'] = total_choices / domain_stats[domain_name]['total']
            
            # Show top question types
            print(f"  Question types:")
            for qtype, count in domain_stats[domain_name]['question_types'].most_common(3):
                print(f"    - {qtype}: {count}")
            
            # Show sample question
            sample_task = AI2DTask(task_id=task_ids[0], adapter=adapter)
            sample_task.setup()
            print(f"  Sample: {sample_task.question[:80]}...")


def test_image_verification(adapter: AI2DAdapter):
    """Verify images exist and analyze their properties"""
    print("\n" + "="*60)
    print("Testing Image Verification")
    print("="*60)
    
    # Sample tasks
    sample_size = 100
    task_ids = adapter.get_task_ids(limit=sample_size, shuffle=True, seed=42)
    
    stats = {
        'total': 0,
        'valid_images': 0,
        'missing_images': 0,
        'image_sizes': Counter(),
        'aspect_ratios': Counter(),
        'errors': []
    }
    
    for task_id in task_ids:
        stats['total'] += 1
        try:
            task_data = adapter.get_task_data(task_id)
            image_path = Path(task_data.get('image_path', ''))
            
            if image_path.exists():
                stats['valid_images'] += 1
                
                # Load image to verify and get properties
                img = Image.open(image_path)
                size = img.size
                stats['image_sizes'][size] += 1
                
                # Calculate aspect ratio
                aspect_ratio = round(size[0] / size[1], 2)
                stats['aspect_ratios'][aspect_ratio] += 1
                
                img.close()
            else:
                stats['missing_images'] += 1
                
        except Exception as e:
            stats['errors'].append(f"{task_id}: {str(e)}")
    
    print(f"\nImage Verification Results:")
    print(f"  - Total checked: {stats['total']}")
    print(f"  - Valid images: {stats['valid_images']} ({stats['valid_images']/stats['total']*100:.1f}%)")
    print(f"  - Missing images: {stats['missing_images']}")
    print(f"  - Unique image sizes: {len(stats['image_sizes'])}")
    
    if stats['image_sizes']:
        print("\n  Common image sizes:")
        for size, count in stats['image_sizes'].most_common(5):
            print(f"    - {size}: {count} images")
    
    if stats['aspect_ratios']:
        print("\n  Common aspect ratios:")
        for ratio, count in stats['aspect_ratios'].most_common(5):
            print(f"    - {ratio}: {count} images")
    
    if stats['errors']:
        print(f"\n  Errors found: {len(stats['errors'])}")
        for error in stats['errors'][:3]:
            print(f"    - {error}")


def visualize_sample_tasks(adapter: AI2DAdapter, n_samples: int = 6):
    """Visualize sample tasks from different question types"""
    print("\n" + "="*60)
    print(f"Visualizing {n_samples} Sample Tasks")
    print("="*60)
    
    # Get samples from different question types
    question_types = ['identification', 'counting', 'comparison', 'spatial', 'process', 'labeling']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    visualized = 0
    for idx, qtype in enumerate(question_types):
        if visualized >= n_samples:
            break
            
        task_ids = adapter.get_task_ids(question_type=qtype, limit=1)
        if not task_ids:
            # Try to get any task
            task_ids = adapter.get_task_ids(limit=1, shuffle=True)
        
        if task_ids:
            try:
                task = AI2DTask(task_id=task_ids[0], adapter=adapter)
                goal, info = task.setup()
                
                # Get image path
                task_data = adapter.get_task_data(task.task_id)
                image_path = Path(task_data.get('image_path', ''))
                
                if image_path.exists():
                    # Load and display image
                    img = Image.open(image_path)
                    axes[idx].imshow(img)
                    axes[idx].axis('off')
                    
                    # Add title with task info
                    title = f"{qtype.capitalize()}\n"
                    title += f"Q: {task.question[:40]}...\n"
                    title += f"A: {task.answer} ({task.answer_letter})"
                    axes[idx].set_title(title, fontsize=9, pad=10)
                    
                    img.close()
                    visualized += 1
                    
                    print(f"\n[Sample {visualized} - {qtype}]")
                    print(f"  Question: {task.question}")
                    print(f"  Choices: {task.choices}")
                    print(f"  Answer: {task.answer}")
                    print(f"  Domain: {info.get('science_domain', 'N/A')}")
                    
            except Exception as e:
                print(f"Error with task {task_ids[0]}: {e}")
                axes[idx].text(0.5, 0.5, f"Error loading\n{qtype}", 
                              ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(visualized, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    sample_file = "ai2d_samples.png"
    plt.savefig(sample_file, dpi=150, bbox_inches='tight')
    print(f"\nSample visualization saved to: {sample_file}")
    plt.close()


def test_complete_workflow(adapter: AI2DAdapter):
    """Test complete VQA workflow with AI2D"""
    print("\n" + "="*60)
    print("Testing Complete VQA Workflow")
    print("="*60)
    
    # Test different question types
    test_types = [
        {'question_type': 'identification', 'desc': 'Identification'},
        {'question_type': 'counting', 'desc': 'Counting'},
        {'question_type': 'comparison', 'desc': 'Comparison'}
    ]
    
    for test_config in test_types:
        print(f"\n[Testing {test_config['desc']} Questions]")
        
        task_ids = adapter.get_task_ids(question_type=test_config['question_type'], limit=1)
        if not task_ids:
            print(f"  No {test_config['question_type']} tasks found")
            continue
            
        task = AI2DTask(task_id=task_ids[0], adapter=adapter)
        goal, info = task.setup()
        
        print(f"  Task Setup:")
        print(f"    - Question: {task.question}")
        print(f"    - Choices: {task.choices}")
        print(f"    - Expected answer: {task.answer} (Letter: {task.answer_letter})")
        print(f"    - Category: {info.get('question_category', 'N/A')}")
        print(f"    - Domain: {info.get('science_domain', 'N/A')}")
        
        # Get observation
        obs = task.get_observation()
        print(f"\n  Observation:")
        print(f"    - Type: {obs['type']}")
        print(f"    - Image available: {obs['vqa_info']['image_path'] is not None}")
        
        # Simulate different VLM responses
        print(f"\n  Simulating VLM responses:")
        
        # 1. Correct answer (letter)
        reward, done, message, val_info = task.validate([], task.answer_letter)
        print(f"    1. Letter '{task.answer_letter}': {message[:60]}...")
        
        # 2. Correct answer (content)
        reward, done, message, val_info = task.validate([], task.answer)
        print(f"    2. Content '{task.answer}': {message[:60]}...")
        
        # 3. Wrong answer
        wrong_letter = 'A' if task.answer_letter != 'A' else 'B'
        reward, done, message, val_info = task.validate([], wrong_letter)
        print(f"    3. Wrong '{wrong_letter}': {message[:60]}...")
        
        # 4. Answer in sentence
        sentence = f"Looking at the diagram, I believe the answer is {task.answer}"
        reward, done, message, val_info = task.validate([], sentence)
        print(f"    4. In sentence: {message[:60]}...")
        
        # Get metrics
        metrics = task.get_metrics()
        print(f"\n  Task Metrics:")
        print(f"    - Complexity: {metrics['question_complexity']}")
        print(f"    - Category: {metrics['question_category']}")
        print(f"    - Choice count: {metrics['num_choices']}")


def test_special_cases(adapter: AI2DAdapter):
    """Test special cases and edge conditions"""
    print("\n" + "="*60)
    print("Testing Special Cases")
    print("="*60)
    
    # Test biology diagrams
    print("\n[Biology Diagram Tasks]")
    bio_tasks = adapter.get_biology_tasks(n=3)
    if bio_tasks:
        for i, task_id in enumerate(bio_tasks[:2]):
            task_data = adapter.get_task_data(task_id)
            print(f"  {i+1}. {task_data['question']}")
            print(f"     Choices: {task_data['choices']}")
    
    # Test physics diagrams
    print(f"\n[Physics Diagram Tasks]")
    physics_tasks = adapter.get_physics_tasks(n=3)
    if physics_tasks:
        for i, task_id in enumerate(physics_tasks[:2]):
            task_data = adapter.get_task_data(task_id)
            print(f"  {i+1}. {task_data['question']}")
    
    # Test earth science diagrams
    print(f"\n[Earth Science Diagram Tasks]")
    earth_tasks = adapter.get_earth_science_tasks(n=3)
    if earth_tasks:
        for i, task_id in enumerate(earth_tasks[:2]):
            task_data = adapter.get_task_data(task_id)
            print(f"  {i+1}. {task_data['question']}")
    
    # Test spatial reasoning
    print(f"\n[Spatial Reasoning Tasks]")
    spatial_tasks = adapter.get_spatial_tasks(n=3)
    if spatial_tasks:
        for i, task_id in enumerate(spatial_tasks[:2]):
            task = AI2DTask(task_id=task_id, adapter=adapter)
            task.setup()
            print(f"  {i+1}. {task.question}")
            print(f"     Type: Requires understanding spatial relationships in diagram")
    
    # Test process/sequence understanding
    print(f"\n[Process Understanding Tasks]")
    process_tasks = adapter.get_process_tasks(n=3)
    if process_tasks:
        for i, task_id in enumerate(process_tasks[:2]):
            task_data = adapter.get_task_data(task_id)
            print(f"  {i+1}. {task_data['question']}")


def test_vlm_integration_example(adapter: AI2DAdapter):
    """Show example of VLM integration"""
    print("\n" + "="*60)
    print("VLM Integration Example")
    print("="*60)
    
    # Get a sample task
    task_ids = adapter.get_task_ids(question_type='identification', limit=1)
    if not task_ids:
        task_ids = adapter.get_task_ids(limit=1)
    
    task_data = adapter.get_task_data(task_ids[0])
    
    print("\nSample VLM Integration:")
    print("-" * 50)
    
    # Build example prompt
    sample_prompt = f"Question: {task_data['question']}\n"
    sample_prompt += f"Choices:\n"
    for i, choice in enumerate(task_data['choices']):
        letter = chr(65 + i)  # A, B, C, D
        sample_prompt += f"{letter}. {choice}\n"
    sample_prompt += "\nPlease analyze the science diagram and select the correct answer."
    
    print(f"Task ID: {task_ids[0]}")
    print(f"Image path: {task_data['image_path']}")
    print(f"\nSample prompt:")
    print(sample_prompt)
    print(f"\nCorrect answer: {task_data['answer']}")
    
    print("\n# Example integration code:")
    print("""
def evaluate_ai2d_with_vlm(adapter, model, processor, num_tasks=100):
    task_ids = adapter.get_task_ids(limit=num_tasks, shuffle=True)
    results = {
        'correct': 0, 
        'total': 0, 
        'by_type': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'by_domain': defaultdict(lambda: {'correct': 0, 'total': 0})
    }
    
    for task_id in task_ids:
        task_data = adapter.get_task_data(task_id)
        
        # Load image
        image = Image.open(task_data['image_path'])
        
        # Build prompt
        prompt = f"Question: {task_data['question']}\\n"
        prompt += "Choices:\\n"
        for i, choice in enumerate(task_data['choices']):
            letter = chr(65 + i)
            prompt += f"{letter}. {choice}\\n"
        prompt += "\\nPlease analyze the science diagram and select the correct answer (A, B, C, or D)."
        
        # Get model prediction
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        outputs = model.generate(**inputs)
        response = processor.decode(outputs[0], skip_special_tokens=True)
        
        # Create task for validation
        task = AI2DTask(task_id=task_id, adapter=adapter)
        success, feedback = task.check_success(response)
        
        # Get task info for categorization
        _, info = task.setup()
        question_type = info.get('question_category', 'unknown')
        science_domain = info.get('science_domain', 'unknown')
        
        # Update results
        results['total'] += 1
        results['by_type'][question_type]['total'] += 1
        results['by_domain'][science_domain]['total'] += 1
        
        if success:
            results['correct'] += 1
            results['by_type'][question_type]['correct'] += 1
            results['by_domain'][science_domain]['correct'] += 1
        
        image.close()
    
    # Calculate accuracy
    overall_accuracy = (results['correct'] / results['total']) * 100
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    
    # By question type
    print("\\nAccuracy by Question Type:")
    for qtype, counts in results['by_type'].items():
        if counts['total'] > 0:
            acc = (counts['correct'] / counts['total']) * 100
            print(f"  {qtype}: {acc:.2f}% ({counts['correct']}/{counts['total']})")
    
    # By science domain
    print("\\nAccuracy by Science Domain:")
    for domain, counts in results['by_domain'].items():
        if counts['total'] > 0:
            acc = (counts['correct'] / counts['total']) * 100
            print(f"  {domain}: {acc:.2f}% ({counts['correct']}/{counts['total']})")
    
    return results
""")


def test_batch_processing(adapter: AI2DAdapter):
    """Test batch processing capabilities"""
    print("\n" + "="*60)
    print("Testing Batch Processing")
    print("="*60)
    
    # Get a batch of task IDs
    batch_size = 10
    task_ids = adapter.get_task_ids(limit=batch_size, shuffle=True)
    
    print(f"\nProcessing batch of {len(task_ids)} tasks...")
    
    # Get batch data
    start_time = time.time()
    batch_data = adapter.get_batch(task_ids)
    batch_time = time.time() - start_time
    
    print(f"Batch loading time: {batch_time:.3f} seconds")
    print(f"Average per task: {batch_time/len(task_ids):.3f} seconds")
    
    # Analyze batch
    batch_stats = {
        'question_lengths': [],
        'choice_counts': [],
        'question_types': Counter(),
        'domains': Counter()
    }
    
    for task_data in batch_data:
        # Create task to get metadata
        task = AI2DTask(task_id=task_data['id'], adapter=adapter)
        _, info = task.setup()
        
        batch_stats['question_lengths'].append(len(task_data['question'].split()))
        batch_stats['choice_counts'].append(len(task_data['choices']))
        batch_stats['question_types'][info.get('question_category', 'unknown')] += 1
        batch_stats['domains'][info.get('science_domain', 'unknown')] += 1
    
    print(f"\nBatch Statistics:")
    print(f"  Average question length: {sum(batch_stats['question_lengths'])/len(batch_stats['question_lengths']):.1f} words")
    print(f"  Average choices: {sum(batch_stats['choice_counts'])/len(batch_stats['choice_counts']):.1f}")
    print(f"  Question types: {dict(batch_stats['question_types'])}")
    print(f"  Science domains: {dict(batch_stats['domains'])}")


def test_data_validation(adapter: AI2DAdapter):
    """Test data validation"""
    print("\n" + "="*60)
    print("Testing Data Validation")
    print("="*60)
    
    print("\nValidating all tasks...")
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


def main():
    """Main test function"""
    print("="*60)
    print("AI2D-VLMGym Integration Test")
    print("="*60)
    
    try:
        # Test 1: Load adapter
        adapter = test_adapter_loading()
        
        # Test 2: Create tasks
        task = test_task_creation(adapter)
        
        # Test 3: Filtering capabilities
        test_filtering_capabilities(adapter)
        
        # Test 4: Answer validation
        test_answer_validation(adapter)
        
        # Test 5: Science domain analysis
        test_science_domain_analysis(adapter)
        
        # Test 6: Image verification
        test_image_verification(adapter)
        
        # Test 7: Visualize samples
        visualize_sample_tasks(adapter, n_samples=6)
        
        # Test 8: Complete workflow
        test_complete_workflow(adapter)
        
        # Test 9: Special cases
        test_special_cases(adapter)
        
        # Test 10: VLM integration example
        test_vlm_integration_example(adapter)
        
        # Test 11: Batch processing
        test_batch_processing(adapter)
        
        # Test 12: Data validation
        test_data_validation(adapter)
        
        print("\n" + "="*60)
        print("✓ All tests completed successfully!")
        print("="*60)
        
        print("\nNext steps:")
        print("1. Check the saved visualization (ai2d_samples.png)")
        print("2. Test with actual VLM models")
        print("3. Analyze performance by question type and science domain")
        print("4. Consider domain-specific prompting strategies")
        print("5. Test multimodal reasoning capabilities on complex diagrams")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())