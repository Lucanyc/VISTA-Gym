#!/usr/bin/env python3
"""
Test script for ScienceQA dataset integration in VLMGym

This script tests:
1. Loading the ScienceQA adapter
2. Reading task data and images
3. Various filtering options
4. Task class functionality
5. Data integrity
"""

import sys
import logging
from pathlib import Path
from PIL import Image
import json

# Add the project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging for detailed debug output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_adapter_loading():
    """Test loading the ScienceQA adapter"""
    print("\n" + "="*80)
    print("TEST 1: Loading ScienceQA Adapter")
    print("="*80)
    
    try:
        from data_adapters.scienceqa_adapter import ScienceQAAdapter, create_scienceqa_adapter
        print("✓ Successfully imported ScienceQA adapter")
    except ImportError as e:
        print(f"✗ Failed to import adapter: {e}")
        return None
    
    # Try different data paths
    possible_data_paths = [
        "/data/wang/meng/GYM-Work/dataset/ScienceQA/reformatted",
        "/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/scienceqa",
        "./data/scienceqa"
    ]
    
    adapter = None
    for data_path in possible_data_paths:
        try:
            print(f"\nTrying data path: {data_path}")
            adapter = ScienceQAAdapter(
                data_root=data_path,
                split="train",
                validate_images=False  # Initially don't validate to see what loads
            )
            
            if adapter.annotations:
                print(f"✓ Successfully loaded adapter with {len(adapter.annotations)} tasks")
                break
            else:
                print(f"✗ No annotations loaded from {data_path}")
                
        except Exception as e:
            print(f"✗ Error with path {data_path}: {e}")
    
    if not adapter or not adapter.annotations:
        print("\n✗ Failed to load any data. Please check data paths.")
        return None
    
    # Show statistics
    print("\nDataset Statistics:")
    stats = adapter.get_statistics()
    print(f"  Total tasks: {stats.get('total', 0)}")
    print(f"  Multiple choice: {stats.get('multiple_choice_tasks', 0)}")
    print(f"  Tasks with hints: {stats.get('tasks_with_hints', 0)}")
    print(f"  Tasks with lectures: {stats.get('tasks_with_lectures', 0)}")
    
    return adapter

def test_task_loading(adapter, num_samples=5):
    """Test loading individual tasks"""
    print("\n" + "="*80)
    print(f"TEST 2: Loading Individual Tasks (sampling {num_samples})")
    print("="*80)
    
    # Get some task IDs
    task_ids = adapter.get_task_ids(limit=num_samples)
    
    for i, task_id in enumerate(task_ids):
        print(f"\n--- Task {i+1}/{num_samples}: {task_id} ---")
        
        try:
            task_data = adapter.get_task_data(task_id)
            
            # Display task information
            print(f"Question: {task_data['question'][:100]}...")
            print(f"Answer: {task_data['answer']}")
            print(f"Question Type: {task_data.get('question_type', 'N/A')}")
            print(f"Has Choices: {'Yes' if task_data.get('choices') else 'No'}")
            if task_data.get('choices'):
                print(f"Choices: {task_data['choices']}")
            
            # Metadata
            metadata = task_data.get('metadata', {})
            print(f"Grade: {metadata.get('grade', 'N/A')}")
            print(f"Subject: {metadata.get('original_subject', 'N/A')}")
            print(f"Topic: {metadata.get('original_topic', 'N/A')}")
            print(f"Skills: {metadata.get('skills', [])}")
            
            # Special features
            print(f"Has Hint: {'Yes' if task_data.get('hint') else 'No'}")
            if task_data.get('hint'):
                print(f"Hint Preview: {task_data['hint'][:100]}...")
            print(f"Has Lecture: {'Yes' if task_data.get('lecture') else 'No'}")
            print(f"Has Solution: {'Yes' if task_data.get('solution') else 'No'}")
            
            # Image information
            image_path = task_data.get('image_path', '')
            print(f"Image Path: {image_path}")
            
            # Try to verify image
            if image_path:
                image_path_obj = Path(image_path)
                if image_path_obj.exists():
                    print(f"✓ Image exists at: {image_path_obj}")
                    
                    # Try to open and get image info
                    try:
                        with Image.open(image_path_obj) as img:
                            print(f"  Image size: {img.size}")
                            print(f"  Image mode: {img.mode}")
                            print(f"  Image format: {img.format}")
                    except Exception as e:
                        print(f"  ✗ Error opening image: {e}")
                else:
                    print(f"✗ Image NOT found at: {image_path_obj}")
                    
                    # Try alternative paths
                    print("  Trying alternative paths:")
                    alt_paths = [
                        Path("/data/wang/meng/GYM-Work/dataset/ScienceQA/reformatted") / image_path,
                        Path("/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/scienceqa") / image_path,
                    ]
                    for alt_path in alt_paths:
                        if alt_path.exists():
                            print(f"  ✓ Found at: {alt_path}")
                            break
                        else:
                            print(f"  ✗ Not at: {alt_path}")
            
        except Exception as e:
            print(f"✗ Error loading task {task_id}: {e}")
            import traceback
            traceback.print_exc()

def test_filtering(adapter):
    """Test various filtering options"""
    print("\n" + "="*80)
    print("TEST 3: Testing Filtering Options")
    print("="*80)
    
    # Test grade filtering
    print("\n--- Grade Level Filtering ---")
    grades = ['elementary school', 'middle school', 'high school']
    for grade in grades:
        task_ids = adapter.get_task_ids(grade=grade, limit=10)
        print(f"{grade}: {len(task_ids)} tasks found")
        if task_ids:
            sample_task = adapter.get_task_data(task_ids[0])
            print(f"  Sample: {sample_task['question'][:60]}...")
    
    # Test subject filtering
    print("\n--- Subject Filtering ---")
    subjects = ['natural science', 'social science', 'language science']
    for subject in subjects:
        task_ids = adapter.get_task_ids(subject=subject, limit=10)
        print(f"{subject}: {len(task_ids)} tasks found")
    
    # Test skill filtering
    print("\n--- Skill Filtering ---")
    # Get available skills
    stats = adapter.get_statistics()
    skill_dist = stats.get('skill_distribution', {})
    for skill, count in list(skill_dist.items())[:5]:  # Top 5 skills
        task_ids = adapter.get_task_ids(skill=skill, limit=5)
        print(f"{skill}: {count} total, sampled {len(task_ids)}")
    
    # Test combined filtering
    print("\n--- Combined Filtering ---")
    task_ids = adapter.get_task_ids(
        grade='elementary school',
        subject='natural science',
        has_choices=True,
        limit=5
    )
    print(f"Elementary + Natural Science + Multiple Choice: {len(task_ids)} tasks")
    
    # Test tasks with special features
    print("\n--- Special Features ---")
    hint_tasks = adapter.get_task_ids(has_hint=True, limit=10)
    lecture_tasks = adapter.get_task_ids(has_lecture=True, limit=10)
    print(f"Tasks with hints: {len(hint_tasks)}")
    print(f"Tasks with lectures: {len(lecture_tasks)}")

def test_task_class():
    """Test the ScienceQA task class"""
    print("\n" + "="*80)
    print("TEST 4: Testing ScienceQA Task Class")
    print("="*80)
    
    try:
        # Add task directory to path
        sys.path.insert(0, str(project_root / "vlm_gym" / "environments"))
        from task.scienceqa import ScienceQATask
        print("✓ Successfully imported ScienceQA task class")
    except ImportError as e:
        print(f"✗ Failed to import task class: {e}")
        return
    
    # Create a mock task
    mock_task_data = {
        'id': 'test_scienceqa_1',
        'question': 'Which state is farthest north?',
        'answer': 'West Virginia',
        'choices': ['West Virginia', 'Louisiana', 'Arizona', 'Oklahoma'],
        'image_path': 'test_image.png',
        'hint': 'Look at the compass rose on the map.',
        'lecture': 'Maps have cardinal directions...',
        'solution': 'To find the answer, look at the compass rose...',
        'metadata': {
            'grade': 'elementary school',
            'original_subject': 'social science',
            'original_topic': 'geography',
            'skills': ['spatial reasoning', 'map reading']
        }
    }
    
    # Initialize task
    task = ScienceQATask(
        task_id='test_1',
        question=mock_task_data['question'],
        answer=mock_task_data['answer'],
        choices=mock_task_data['choices'],
        image_path=mock_task_data['image_path']
    )
    
    # Set additional attributes
    task.hint = mock_task_data['hint']
    task.lecture = mock_task_data['lecture']
    task.solution = mock_task_data['solution']
    task.metadata = mock_task_data['metadata']
    
    # Test setup
    print("\n--- Testing Task Setup ---")
    goal, info = task.setup()
    print(f"Task Goal Preview: {goal[:200]}...")
    print(f"Task Info Keys: {list(info.keys())}")
    print(f"Subject: {info.get('subject')}")
    print(f"Grade Level: {info.get('grade_level')}")
    print(f"Skills: {info.get('skills')}")
    
    # Test answer checking
    print("\n--- Testing Answer Checking ---")
    test_answers = [
        'West Virginia',  # Correct
        'west virginia',  # Correct (case insensitive)
        'A',             # Should work if A is correct
        'Louisiana',     # Incorrect
        'Texas'          # Not in choices
    ]
    
    for test_answer in test_answers:
        success, feedback = task.check_success(test_answer)
        print(f"Answer '{test_answer}': {success} - {feedback}")
    
    # Test metrics
    print("\n--- Testing Metrics ---")
    metrics = task.get_metrics()
    print(f"Metrics: {json.dumps(metrics, indent=2)}")

def test_data_validation(adapter):
    """Test data validation"""
    print("\n" + "="*80)
    print("TEST 5: Data Validation")
    print("="*80)
    
    print("\nValidating all tasks...")
    issues = adapter.validate_all()
    
    if not issues:
        print("✓ No validation issues found!")
    else:
        print(f"✗ Found {len(issues)} types of issues:")
        for issue_type, task_ids in issues.items():
            print(f"  {issue_type}: {len(task_ids)} tasks")
            if len(task_ids) <= 3:
                print(f"    Tasks: {task_ids}")
            else:
                print(f"    First 3 tasks: {task_ids[:3]}")

def main():
    """Run all tests"""
    print("\n" + "#"*80)
    print("# ScienceQA Integration Test Suite")
    print("#"*80)
    
    # Test 1: Load adapter
    adapter = test_adapter_loading()
    if not adapter:
        print("\n✗ Cannot proceed without adapter. Exiting.")
        return
    
    # Test 2: Load individual tasks
    test_task_loading(adapter, num_samples=3)
    
    # Test 3: Test filtering
    test_filtering(adapter)
    
    # Test 4: Test task class
    test_task_class()
    
    # Test 5: Data validation
    test_data_validation(adapter)
    
    print("\n" + "#"*80)
    print("# Test Suite Complete")
    print("#"*80)
    
    # Summary
    print("\nSummary:")
    stats = adapter.get_statistics()
    print(f"  Total tasks loaded: {stats.get('total', 0)}")
    print(f"  Unique images: {stats.get('unique_images', 0)}")
    print(f"  Grade distribution: {len(adapter.grade_index)} grades")
    print(f"  Subject distribution: {len(adapter.subject_index)} subjects")
    print(f"  Skill distribution: {len(adapter.skill_index)} skills")

if __name__ == "__main__":
    main()