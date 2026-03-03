"""
Test script for PMC-VQA integration with VLMGym
Tests medical visual question answering with multiple choice questions
Quick test version - uses only 20 data points for fast testing
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
from vlm_gym.environments.task.pmcvqa import PMCVQATask
from vlm_gym.environments.task.vision_qa_task import VisionQATask
from data_adapters.pmcvqa_adapter import PMCVQAAdapter

# Global constant for quick testing
QUICK_TEST_SIZE = 20


def test_adapter_loading():
    """Test PMC-VQA adapter initialization and data loading"""
    print("\n" + "="*60)
    print("Testing PMC-VQA Adapter Loading")
    print("="*60)
    
    # Initialize adapter (don't validate all images for speed)
    adapter = PMCVQAAdapter(
        data_root="/data/wang/meng/GYM-Work/dataset/PMC-VQA",
        annotation_files="/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/PMC-VQA/train/pmc_vqa_train.json",
        validate_images=False  # Skip validation for speed
    )
    
    # Get statistics
    stats = adapter.get_statistics()
    print(f"\n✓ Adapter loaded successfully!")
    print(f"  - Total tasks: {stats['total']:,}")
    print(f"  - Unique images: {stats.get('unique_images', 'N/A'):,}")
    print(f"  - Average questions per image: {stats.get('avg_questions_per_image', 0):.1f}")
    
    # Split distribution
    print(f"\n  Split Distribution:")
    for split, count in sorted(stats.get('split_distribution', {}).items()):
        percentage = (count / stats['total']) * 100
        print(f"    - {split}: {count:,} ({percentage:.1f}%)")
    
    # Question type distribution
    print(f"\n  Question Type Distribution:")
    for qtype, count in sorted(stats.get('question_type_distribution', {}).items()):
        percentage = (count / stats['total']) * 100
        print(f"    - {qtype}: {count:,} ({percentage:.1f}%)")
    
    return adapter


def test_task_creation(adapter: PMCVQAAdapter):
    """Test creating PMC-VQA tasks"""
    print("\n" + "="*60)
    print("Testing Task Creation")
    print("="*60)
    
    # Get a few sample tasks
    task_ids = adapter.get_task_ids(limit=3)
    
    for idx, task_id in enumerate(task_ids, 1):
        task = PMCVQATask(task_id=task_id, adapter=adapter)
        goal, info = task.setup()
        
        print(f"\n[Sample Task {idx}]")
        print(f"  - Task ID: {task_id}")
        print(f"  - Question: {task.question}")
        print(f"  - Choices:")
        for i, choice in enumerate(task.choices):
            letter = chr(ord('A') + i)
            is_correct = letter == task.metadata.get('ground_truth_answer', '')
            marker = " ✓" if is_correct else ""
            print(f"    {letter}. {choice}{marker}")
        print(f"  - Medical Domain: {info['medical_domain']}")
        print(f"  - Imaging Modality: {info['imaging_modality']}")
        print(f"  - Requires Localization: {info['requires_localization']}")
    
    return task if 'task' in locals() else None


def test_medical_domain_detection(adapter: PMCVQAAdapter):
    """Test medical domain and modality detection"""
    print("\n" + "="*60)
    print("Testing Medical Domain Detection")
    print("="*60)
    
    # Sample tasks for analysis - REDUCED TO 20
    sample_size = QUICK_TEST_SIZE
    task_ids = adapter.get_task_ids(limit=sample_size, shuffle=True, seed=42)
    
    domain_stats = Counter()
    modality_stats = Counter()
    localization_count = 0
    
    for task_id in task_ids:
        task = PMCVQATask(task_id=task_id, adapter=adapter)
        goal, info = task.setup()
        
        domain_stats[info['medical_domain']] += 1
        modality_stats[info['imaging_modality']] += 1
        if info['requires_localization']:
            localization_count += 1
    
    print(f"\nAnalysis of {sample_size} sampled tasks (quick test):")
    
    print("\nMedical Domains:")
    for domain, count in domain_stats.most_common():
        percentage = (count / sample_size) * 100
        print(f"  - {domain}: {count} ({percentage:.1f}%)")
    
    print("\nImaging Modalities:")
    for modality, count in modality_stats.most_common():
        percentage = (count / sample_size) * 100
        print(f"  - {modality}: {count} ({percentage:.1f}%)")
    
    print(f"\nLocalization Requirements:")
    loc_percentage = (localization_count / sample_size) * 100
    print(f"  - Tasks requiring localization: {localization_count} ({loc_percentage:.1f}%)")


def test_answer_validation(adapter: PMCVQAAdapter):
    """Test answer validation with various formats"""
    print("\n" + "="*60)
    print("Testing Answer Validation (Multiple Choice)")
    print("="*60)
    
    # Get a sample task
    task_ids = adapter.get_task_ids(limit=1)
    if not task_ids:
        print("No tasks available")
        return
        
    task = PMCVQATask(task_id=task_ids[0], adapter=adapter)
    task.setup()
    
    print(f"Question: {task.question}")
    print(f"Choices:")
    for i, choice in enumerate(task.choices):
        letter = chr(ord('A') + i)
        print(f"  {letter}. {choice}")
    
    correct_letter = task.metadata.get('ground_truth_answer', 'A')
    print(f"\nCorrect answer: {correct_letter}")
    
    # Test cases for multiple choice answers
    wrong_letter = 'B' if correct_letter != 'B' else 'C'
    
    test_cases = [
        # Correct answers (should pass)
        (correct_letter, "Single letter correct answer ✓"),
        (correct_letter.lower(), "Lowercase correct answer ✓"),
        (f"The answer is {correct_letter}", "Answer in sentence ✓"),
        (f"{correct_letter}.", "Answer with period ✓"),
        (f"Option {correct_letter}", "Answer with 'Option' prefix ✓"),
        
        # Wrong answers (should fail)
        (wrong_letter, "Wrong letter answer ✗"),
        ("", "Empty answer ✗"),
        ("E", "Invalid option (E) ✗"),
        ("AB", "Multiple letters ✗"),
        ("Yes", "Yes/No answer (wrong format) ✗"),
        ("All of the above", "Text answer instead of letter ✗"),
    ]
    
    # If we have the actual choice text, test that too
    if task.choices and len(task.choices) > ord(correct_letter) - ord('A'):
        correct_choice_text = task.choices[ord(correct_letter) - ord('A')]
        test_cases.append((correct_choice_text, "Full choice text ✓"))
    
    print("\nTesting various answer formats:")
    print("(✓ = should pass, ✗ = should fail)")
    
    for answer, description in test_cases:
        success, feedback = task.check_success(answer)
        expected = "✓" in description
        actual = "✓" if success else "✗"
        match = "✅" if (expected == success) else "❌"
        print(f"  {match} {description}: '{answer}' -> {feedback[:50]}...")


def test_filtering_capabilities(adapter: PMCVQAAdapter):
    """Test various filtering options"""
    print("\n" + "="*60)
    print("Testing Filtering Capabilities")
    print("="*60)
    
    # Test split filtering with limit for quick test
    print("\n[Split Filtering]")
    train_tasks = adapter.get_task_ids(split='train', limit=10)
    print(f"  - Train tasks (limited sample): {len(train_tasks):,}")
    
    # Test question type filtering with limit
    print("\n[Question Type Filtering]")
    qtype = 'multiple_choice'  # Default for PMC-VQA
    mc_tasks = adapter.get_task_ids(question_type=qtype, limit=10)
    print(f"  - Multiple choice tasks (limited sample): {len(mc_tasks):,}")
    
    # Test medical specialty examples - quick version
    print("\n[Medical Examples]")
    # Get a small sample to find medical examples
    sample_tasks = adapter.get_task_ids(limit=10, shuffle=True, seed=42)
    medical_keywords = ['uptake', 'lesion', 'tumor', 'diagnosis', 'radiological']
    
    medical_examples = []
    for task_id in sample_tasks:
        if len(medical_examples) >= 5:
            break
        task_data = adapter.get_task_data(task_id)
        question = task_data.get('question', '').lower()
        if any(keyword in question for keyword in medical_keywords):
            medical_examples.append(task_id)
    
    print(f"  Found {len(medical_examples)} medical-focused examples:")
    for idx, task_id in enumerate(medical_examples[:3], 1):
        task_data = adapter.get_task_data(task_id)
        print(f"    {idx}. {task_data['question'][:60]}...")


def test_image_verification(adapter: PMCVQAAdapter):
    """Verify images exist and analyze their properties"""
    print("\n" + "="*60)
    print("Testing Image Verification")
    print("="*60)
    
    # Sample tasks - REDUCED TO 20
    sample_size = QUICK_TEST_SIZE
    task_ids = adapter.get_task_ids(limit=sample_size, shuffle=True, seed=42)
    
    stats = {
        'total': 0,
        'valid_images': 0,
        'missing_images': 0,
        'image_sizes': Counter(),
        'unique_images': set(),
        'errors': []
    }
    
    for task_id in task_ids:
        stats['total'] += 1
        try:
            task_data = adapter.get_task_data(task_id)
            image_path = Path(task_data.get('image_path', ''))
            
            # Track unique images
            stats['unique_images'].add(str(image_path))
            
            if image_path.exists():
                stats['valid_images'] += 1
                
                # Load image to verify
                img = Image.open(image_path)
                stats['image_sizes'][img.size] += 1
                img.close()
            else:
                stats['missing_images'] += 1
                
        except Exception as e:
            stats['errors'].append(f"{task_id}: {str(e)}")
    
    print(f"\nImage Verification Results (quick test):")
    print(f"  - Total checked: {stats['total']}")
    print(f"  - Valid images: {stats['valid_images']} ({stats['valid_images']/stats['total']*100:.1f}%)")
    print(f"  - Missing images: {stats['missing_images']}")
    print(f"  - Unique images in sample: {len(stats['unique_images'])}")
    
    if stats['image_sizes']:
        print("\n  Common image sizes:")
        for size, count in stats['image_sizes'].most_common(3):
            print(f"    - {size}: {count} images")


def visualize_sample_tasks(adapter: PMCVQAAdapter, n_samples: int = 6):
    """Visualize sample medical VQA tasks"""
    print("\n" + "="*60)
    print(f"Visualizing {n_samples} Sample Medical VQA Tasks")
    print("="*60)
    
    # Get sample tasks
    task_ids = adapter.get_task_ids(limit=n_samples, shuffle=True, seed=42)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, task_id in enumerate(task_ids[:n_samples]):
        try:
            task = PMCVQATask(task_id=task_id, adapter=adapter)
            goal, info = task.setup()
            
            image_path = Path(task.image_path) if task.image_path else None
            if image_path and image_path.exists():
                # Load and display image
                img = Image.open(image_path)
                axes[idx].imshow(img)
                axes[idx].axis('off')
                
                # Add title with question info
                title = f"{info['medical_domain'].replace('_', ' ').title()}\n"
                title += f"Q: {task.question[:40]}...\n"
                correct_letter = task.metadata.get('ground_truth_answer', '?')
                title += f"A: {correct_letter}"
                axes[idx].set_title(title, fontsize=10, pad=10)
                
                img.close()
                
                print(f"\n[Sample {idx+1}]")
                print(f"  Domain: {info['medical_domain']}")
                print(f"  Question: {task.question[:80]}...")
                print(f"  Answer: {correct_letter}")
                
        except Exception as e:
            print(f"Error with {task_id}: {e}")
            axes[idx].text(0.5, 0.5, f"Error loading\ntask {idx+1}", 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].axis('off')
    
    plt.tight_layout()
    sample_file = "pmcvqa_samples.png"
    plt.savefig(sample_file, dpi=150, bbox_inches='tight')
    print(f"\nSample visualization saved to: {sample_file}")
    plt.close()


def test_complete_workflow(adapter: PMCVQAAdapter):
    """Test complete VQA workflow with PMC-VQA"""
    print("\n" + "="*60)
    print("Testing Complete VQA Workflow")
    print("="*60)
    
    # Get a task that requires localization and one that doesn't - LIMIT TO 20
    all_tasks = adapter.get_task_ids(limit=QUICK_TEST_SIZE)
    
    localization_task = None
    general_task = None
    
    for task_id in all_tasks:
        task = PMCVQATask(task_id=task_id, adapter=adapter)
        goal, info = task.setup()
        
        if info['requires_localization'] and not localization_task:
            localization_task = (task, info)
        elif not info['requires_localization'] and not general_task:
            general_task = (task, info)
        
        if localization_task and general_task:
            break
    
    # Test both types of tasks
    for task_type, task_info in [("General", general_task), ("Localization", localization_task)]:
        if not task_info:
            print(f"\n[No {task_type} task found in quick sample]")
            continue
            
        task, info = task_info
        print(f"\n[Testing {task_type} Medical VQA Task]")
        
        print(f"  Task Setup:")
        print(f"    - Question: {task.question}")
        print(f"    - Medical Domain: {info['medical_domain']}")
        print(f"    - Requires Localization: {info['requires_localization']}")
        
        # Get observation
        obs = task.get_observation()
        print(f"\n  Observation:")
        print(f"    - Type: {obs['type']}")
        print(f"    - Image available: {obs['vqa_info']['image_path'] is not None}")
        
        # Simulate different VLM responses
        correct_letter = task.metadata.get('ground_truth_answer', 'A')
        wrong_letter = 'B' if correct_letter != 'B' else 'C'
        
        print(f"\n  Simulating VLM responses:")
        
        # 1. Correct answer
        reward, done, message, val_info = task.validate([], correct_letter)
        print(f"    1. Correct '{correct_letter}': {message[:60]}...")
        
        # 2. Wrong answer
        reward, done, message, val_info = task.validate([], wrong_letter)
        print(f"    2. Wrong '{wrong_letter}': {message[:60]}...")
        
        # 3. Full text answer
        if task.choices and len(task.choices) > ord(correct_letter) - ord('A'):
            full_answer = task.choices[ord(correct_letter) - ord('A')]
            reward, done, message, val_info = task.validate([], full_answer)
            print(f"    3. Full text '{full_answer[:30]}...': {message[:60]}...")
        
        # Get metrics
        metrics = task.get_metrics()
        print(f"\n  Task Metrics:")
        print(f"    - Question complexity: {metrics['question_complexity']}")
        print(f"    - Imaging modality: {metrics['imaging_modality']}")


def test_sampling_capabilities(adapter: PMCVQAAdapter):
    """Test sampling capabilities"""
    print("\n" + "="*60)
    print("Testing Sampling Capabilities")
    print("="*60)
    
    # Get available task count
    all_task_ids = adapter.get_task_ids(limit=10)  # Just check first 1000
    available_count = len(all_task_ids)
    
    # Test random sampling - reduced size
    print("\n[Random Sampling]")
    sample_size = min(10, available_count // 2)
    random_ids = adapter.sample_tasks(n=sample_size, seed=42)
    print(f"  Sampled {len(random_ids)} random tasks")
    
    # Test stratified sampling by split - reduced size
    print("\n[Stratified Sampling by Split]")
    stratified_size = 20
    stratified_ids = adapter.sample_tasks(n=stratified_size, stratified=True, seed=42)
    
    # Check distribution
    split_counts = Counter()
    for task_id in stratified_ids:
        task_data = adapter.get_task_data(task_id)
        split = task_data.get('metadata', {}).get('split', 'unknown')
        split_counts[split] += 1
    
    print(f"  Sampled {len(stratified_ids)} tasks:")
    for split, count in split_counts.most_common():
        print(f"    - {split}: {count} ({count/len(stratified_ids)*100:.1f}%)")


def test_tool_integration_hints(adapter: PMCVQAAdapter):
    """Test tool integration hints and guidance"""
    print("\n" + "="*60)
    print("Testing Tool Integration Hints")
    print("="*60)
    
    # Find tasks that might benefit from different tools - LIMIT TO 10
    sample_size = 10
    task_ids = adapter.get_task_ids(limit=sample_size)
    
    tool_suggestions = {
        'grounding_dino': [],
        'deepeyes': [],
        'general': []
    }
    
    for task_id in task_ids:
        task = PMCVQATask(task_id=task_id, adapter=adapter)
        goal, info = task.setup()
        
        if info['requires_localization']:
            tool_suggestions['grounding_dino'].append({
                'task_id': task_id,
                'question': task.question[:60] + '...',
                'hint': task.get_hint()
            })
        elif 'detail' in task.question.lower() or 'zoom' in task.question.lower():
            tool_suggestions['deepeyes'].append({
                'task_id': task_id,
                'question': task.question[:60] + '...',
                'hint': task.get_hint()
            })
        else:
            tool_suggestions['general'].append({
                'task_id': task_id,
                'question': task.question[:60] + '...',
                'hint': task.get_hint()
            })
    
    print("\nTool Suggestions Analysis (quick test):")
    
    for tool, tasks in tool_suggestions.items():
        if tasks:
            print(f"\n[{tool.replace('_', ' ').title()} Suggested Tasks]")
            for task_info in tasks[:2]:  # Show first 2 examples
                print(f"  Q: {task_info['question']}")
                print(f"  Hint: {task_info['hint']}")
                print()


def test_vlm_integration_example(adapter: PMCVQAAdapter):
    """Show example of VLM integration with medical VQA"""
    print("\n" + "="*60)
    print("VLM Integration Example for Medical VQA")
    print("="*60)
    
    # Get a sample task
    task_ids = adapter.get_task_ids(limit=1)
    task_data = adapter.get_task_data(task_ids[0])
    
    print("\nSample VLM Integration for Medical Images:")
    print("-" * 50)
    
    # Build example prompt
    sample_prompt = f"Medical Image Analysis Task:\n\n"
    sample_prompt += f"Question: {task_data['question']}\n\n"
    sample_prompt += "Choices:\n"
    for i, choice in enumerate(task_data['choices']):
        letter = chr(ord('A') + i)
        sample_prompt += f"  {letter}. {choice}\n"
    sample_prompt += "\nPlease select the correct answer (A, B, C, or D)."
    
    print(f"Task ID: {task_ids[0]}")
    print(f"Image path: {task_data['image_path']}")
    print(f"\nSample prompt:")
    print(sample_prompt)
    
    correct_letter = task_data['metadata']['ground_truth_answer']
    correct_choice = task_data['choices'][ord(correct_letter) - ord('A')]
    print(f"\nCorrect answer: {correct_letter} - {correct_choice}")
    
    print("\n# Example integration code provided above can be used with actual VLM models")


def main():
    """Main test function"""
    print("="*60)
    print("PMC-VQA - VLMGym Integration Test")
    print("*** QUICK TEST VERSION - Using only 20 data points ***")
    print("="*60)
    
    try:
        # Test 1: Load adapter
        print("\n[1/11] Loading adapter...")
        adapter = test_adapter_loading()
        
        # Test 2: Create tasks
        print("\n[2/11] Testing task creation...")
        task = test_task_creation(adapter)
        
        # Test 3: Medical domain detection
        print("\n[3/11] Testing medical domain detection...")
        test_medical_domain_detection(adapter)
        
        # Test 4: Answer validation
        print("\n[4/11] Testing answer validation...")
        test_answer_validation(adapter)
        
        # Test 5: Filtering capabilities
        print("\n[5/11] Testing filtering capabilities...")
        test_filtering_capabilities(adapter)
        
        # Test 6: Image verification
        print("\n[6/11] Testing image verification...")
        test_image_verification(adapter)
        
        # Test 7: Visualize samples
        print("\n[7/11] Creating visualization...")
        visualize_sample_tasks(adapter, n_samples=6)
        
        # Test 8: Complete workflow
        print("\n[8/11] Testing complete workflow...")
        test_complete_workflow(adapter)
        
        # Test 9: Sampling capabilities
        print("\n[9/11] Testing sampling capabilities...")
        test_sampling_capabilities(adapter)
        
        # Test 10: Tool integration hints
        print("\n[10/11] Testing tool integration hints...")
        test_tool_integration_hints(adapter)
        
        # Test 11: VLM integration example
        print("\n[11/11] Showing VLM integration example...")
        test_vlm_integration_example(adapter)
        
        print("\n" + "="*60)
        print("✓ All tests completed successfully!")
        print("="*60)
        
        print("\n*** This was a QUICK TEST with limited data points ***")
        print("For full testing, modify QUICK_TEST_SIZE in the script")
        
        print("\nNext steps:")
        print("1. Check the saved visualization (pmcvqa_samples.png)")
        print("2. Review medical domain and modality detection")
        print("3. Test with actual VLM models (Qwen-VL, GPT-4V, etc.)")
        print("4. Analyze performance by medical specialty")
        print("5. Evaluate tool usage effectiveness for localization tasks")
        print("6. Consider fine-tuning on medical terminology if needed")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())