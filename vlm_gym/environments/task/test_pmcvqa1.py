"""
Test script for PMC-VQA integration with VLMGym
Tests medical visual question answering with multiple choice questions
Limited version - only loads 50 tasks for quick testing
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
import logging

# Add VLMGym to path
sys.path.insert(0, '/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista')

# Import required components
from vlm_gym.environments.task.pmcvqa import PMCVQATask
from vlm_gym.environments.task.vision_qa_task import VisionQATask
from data_adapters.pmcvqa_adapter import PMCVQAAdapter

# Global constant for quick testing
QUICK_TEST_SIZE = 10


def test_adapter_loading():
    """Test PMC-VQA adapter initialization and data loading"""
    print("\n" + "="*60)
    print("Testing PMC-VQA Adapter Loading (LIMITED VERSION)")
    print("="*60)
    
    # Create a modified adapter class that limits loading
    class LimitedPMCVQAAdapter(PMCVQAAdapter):
        def _load_annotations(self):
            """Modified load function that only loads 50 tasks"""
            logger = logging.getLogger(__name__)
            logger.info(f"Loading PMC-VQA annotations (LIMITED TO 50 TASKS)")
            
            for ann_file in self.annotation_files:
                if not ann_file.exists():
                    logger.warning(f"Annotation file not found: {ann_file}")
                    continue
                    
                try:
                    with open(ann_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # LIMIT TO FIRST 50 TASKS
                    task_count = 0
                    for item in data:
                        if task_count >= 50:  # Only load 50 tasks
                            break
                        
                        # Ensure it's a PMC-VQA task
                        if item.get('dataset', '').lower() != 'pmc_vqa':
                            continue
                        
                        # Add to collections
                        task_id = item['id']
                        self.annotations.append(item)
                        self._task_index[task_id] = item
                        
                        # Index by metadata
                        metadata = item.get('metadata', {})
                        
                        # Question type indexing
                        q_type = metadata.get('question_type', 'multiple_choice')
                        if q_type not in self.question_types:
                            self.question_types[q_type] = []
                        self.question_types[q_type].append(task_id)
                        
                        # Split indexing
                        split = metadata.get('split', 'train')
                        if split not in self.split_index:
                            self.split_index[split] = []
                        self.split_index[split].append(task_id)
                        
                        # Image indexing
                        img_path = item.get('image_path', '')
                        if img_path not in self.image_index:
                            self.image_index[img_path] = []
                        self.image_index[img_path].append(task_id)
                        
                        # Update stats
                        self.stats['total'] += 1
                        self.stats[f'source_{ann_file.stem}'] += 1
                        self.stats[f'split_{split}'] += 1
                        self.stats[f'qtype_{q_type}'] += 1
                        
                        task_count += 1
                        
                    logger.info(f"Loaded {task_count} tasks from {ann_file.name}")
                    
                except Exception as e:
                    logger.error(f"Error loading {ann_file}: {e}")
                    raise
            
            # Calculate additional statistics
            if self.image_index:
                image_reuse = [len(tasks) for tasks in self.image_index.values()]
                self.stats['unique_images'] = len(self.image_index)
                self.stats['avg_questions_per_image'] = sum(image_reuse) / len(image_reuse)
                self.stats['max_questions_per_image'] = max(image_reuse)
            else:
                self.stats['unique_images'] = 0
                self.stats['avg_questions_per_image'] = 0
                self.stats['max_questions_per_image'] = 0
            
            logger.info(f"Total PMC-VQA tasks loaded: {self.stats['total']} (LIMITED)")
            self._log_statistics()
    
    # Initialize limited adapter
    adapter = LimitedPMCVQAAdapter(
        data_root="/data/wang/meng/GYM-Work/dataset/PMC-VQA",
        annotation_files="/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/PMC-VQA/train/pmc_vqa_train.json",
        validate_images=False  # Skip validation for speed
    )
    
    # Get statistics
    stats = adapter.get_statistics()
    print(f"\n✓ Adapter loaded successfully (LIMITED MODE)!")
    print(f"  - Total tasks: {stats['total']:,}")
    print(f"  - Unique images: {stats.get('unique_images', 'N/A'):,}")
    print(f"  - Average questions per image: {stats.get('avg_questions_per_image', 0):.1f}")
    
    # Split distribution
    print(f"\n  Split Distribution:")
    split_dist = stats.get('split_distribution', {})
    if not split_dist:
        # Calculate from stats if not available
        for key, value in stats.items():
            if key.startswith('split_'):
                split_name = key.replace('split_', '')
                split_dist[split_name] = value
    
    for split, count in sorted(split_dist.items()):
        percentage = (count / stats['total']) * 100 if stats['total'] > 0 else 0
        print(f"    - {split}: {count:,} ({percentage:.1f}%)")
    
    # Question type distribution
    print(f"\n  Question Type Distribution:")
    qtype_dist = stats.get('question_type_distribution', {})
    if not qtype_dist:
        # Calculate from stats if not available
        for key, value in stats.items():
            if key.startswith('qtype_'):
                qtype_name = key.replace('qtype_', '')
                qtype_dist[qtype_name] = value
    
    for qtype, count in sorted(qtype_dist.items()):
        percentage = (count / stats['total']) * 100 if stats['total'] > 0 else 0
        print(f"    - {qtype}: {count:,} ({percentage:.1f}%)")
    
    return adapter


def test_task_creation(adapter):
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


def test_medical_domain_detection(adapter):
    """Test medical domain and modality detection"""
    print("\n" + "="*60)
    print("Testing Medical Domain Detection")
    print("="*60)
    
    # Sample tasks for analysis
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
    
    print(f"\nAnalysis of {sample_size} sampled tasks:")
    
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


def test_answer_validation(adapter):
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
    
    # Test cases
    wrong_letter = 'B' if correct_letter != 'B' else 'C'
    
    test_cases = [
        (correct_letter, "Single letter correct answer", True),
        (correct_letter.lower(), "Lowercase correct answer", True),
        (wrong_letter, "Wrong letter answer", False),
        ("", "Empty answer", False),
        ("E", "Invalid option", False),
    ]
    
    print("\nTesting answer formats:")
    for answer, description, should_pass in test_cases:
        success, feedback = task.check_success(answer)
        result = "✅" if (success == should_pass) else "❌"
        print(f"  {result} {description}: '{answer}' -> {feedback[:50]}...")


def test_filtering_capabilities(adapter):
    """Test various filtering options"""
    print("\n" + "="*60)
    print("Testing Filtering Capabilities")
    print("="*60)
    
    # Test basic filtering
    all_tasks = adapter.get_task_ids()
    print(f"\n[Task Overview]")
    print(f"  - Total available tasks: {len(all_tasks)}")
    
    # Get sample tasks
    sample_tasks = adapter.get_task_ids(limit=5, shuffle=True, seed=42)
    print(f"  - Sample of 5 tasks: {sample_tasks}")
    
    # Show a few example questions
    print("\n[Sample Questions]")
    for i, task_id in enumerate(sample_tasks[:3], 1):
        task_data = adapter.get_task_data(task_id)
        print(f"  {i}. {task_data['question'][:60]}...")


def test_image_verification(adapter):
    """Verify images exist and analyze their properties"""
    print("\n" + "="*60)
    print("Testing Image Verification")
    print("="*60)
    
    # Sample tasks
    sample_size = QUICK_TEST_SIZE
    task_ids = adapter.get_task_ids(limit=sample_size, shuffle=True, seed=42)
    
    stats = {
        'total': 0,
        'valid_images': 0,
        'missing_images': 0,
        'unique_images': set(),
    }
    
    for task_id in task_ids:
        stats['total'] += 1
        try:
            task_data = adapter.get_task_data(task_id)
            image_path = Path(task_data.get('image_path', ''))
            stats['unique_images'].add(str(image_path))
            
            if image_path.exists():
                stats['valid_images'] += 1
            else:
                stats['missing_images'] += 1
                
        except Exception as e:
            print(f"Error checking {task_id}: {e}")
    
    print(f"\nImage Verification Results:")
    print(f"  - Total checked: {stats['total']}")
    print(f"  - Valid images: {stats['valid_images']}")
    print(f"  - Missing images: {stats['missing_images']}")
    print(f"  - Unique images: {len(stats['unique_images'])}")


def test_visualization(adapter):
    """Visualize sample tasks"""
    print("\n" + "="*60)
    print("Creating Visualization")
    print("="*60)
    
    # Get 6 sample tasks
    task_ids = adapter.get_task_ids(limit=6, shuffle=True, seed=42)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, task_id in enumerate(task_ids):
        try:
            task = PMCVQATask(task_id=task_id, adapter=adapter)
            goal, info = task.setup()
            
            ax = axes[idx]
            
            # Try to load image
            image_path = Path(task.image_path) if task.image_path else None
            if image_path and image_path.exists():
                img = Image.open(image_path)
                ax.imshow(img)
                img.close()
            else:
                ax.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center')
                ax.set_facecolor('lightgray')
            
            ax.axis('off')
            
            # Add title
            title = f"{info['medical_domain']}\n"
            title += f"Q: {task.question[:30]}...\n"
            title += f"A: {task.metadata.get('ground_truth_answer', '?')}"
            ax.set_title(title, fontsize=8, pad=5)
            
        except Exception as e:
            axes[idx].text(0.5, 0.5, f'Error\nLoading', ha='center', va='center')
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('pmcvqa_samples.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Visualization saved to pmcvqa_samples.png")


def main():
    """Main test function"""
    print("="*60)
    print("PMC-VQA - VLMGym Integration Test")
    print("*** LIMITED TEST VERSION - Loading only 50 tasks ***")
    print("="*60)
    
    try:
        # Load adapter
        print("\n[1/7] Loading adapter...")
        adapter = test_adapter_loading()
        
        # Test task creation
        print("\n[2/7] Testing task creation...")
        test_task_creation(adapter)
        
        # Test medical domain detection
        print("\n[3/7] Testing medical domain detection...")
        test_medical_domain_detection(adapter)
        
        # Test answer validation
        print("\n[4/7] Testing answer validation...")
        test_answer_validation(adapter)
        
        # Test filtering
        print("\n[5/7] Testing filtering capabilities...")
        test_filtering_capabilities(adapter)
        
        # Test images
        print("\n[6/7] Testing image verification...")
        test_image_verification(adapter)
        
        # Create visualization
        print("\n[7/7] Creating visualization...")
        test_visualization(adapter)
        
        print("\n" + "="*60)
        print("✓ All tests completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())