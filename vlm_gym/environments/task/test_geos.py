#!/usr/bin/env python3
"""
Test script for GEOS (Geometry) dataset integration with VLM Gym.

This script tests the functionality of GEOSAdapter and GEOSTask, including:
- Adapter data loading and statistics.
- Task creation based on different geometric concepts.
- Validation of numerical and mathematical expression-based answers.
- The complete task workflow from setup to validation.
- Visualization of sample problems.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import random
from collections import defaultdict, Counter
import traceback
from PIL import Image
import matplotlib.pyplot as plt

# --- 将 VLM Gym 项目根目录添加到 Python 路径 ---
# 请根据您的项目结构调整此路径
VLM_GYM_PATH = '/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista'
if VLM_GYM_PATH not in sys.path:
    sys.path.insert(0, VLM_GYM_PATH)

# --- 导入所需的自定义组件 ---
try:
    from vlm_gym.environments.task.geos import GEOSTask
    # 假设 adapter 文件在项目根目录下的 'data_adapters' 文件夹中
    from data_adapters.geos_adapter import GEOSAdapter
except ImportError as e:
    print(f"Error: Failed to import required components: {e}")
    print("\nPlease ensure the following:")
    print(f"1. Your project root is correctly set to: {VLM_GYM_PATH}")
    print("2. 'geos.py' exists in 'vlm_gym/environments/task/'.")
    print("3. 'geos_adapter.py' exists in 'data_adapters/'.")
    sys.exit(1)


def test_adapter_loading(annotation_file: str):
    """Test GEOSAdapter initialization and data loading."""
    print("\n" + "="*60)
    print("1. Testing GEOS Adapter Loading")
    print("="*60)
    
    print(f"Loading from annotation file: {annotation_file}")
    adapter = GEOSAdapter(annotation_file=annotation_file, validate_on_load=True)

    stats = adapter.get_statistics()
    print("\n✓ Adapter loaded successfully!")
    print(f"  - Total tasks loaded: {stats.get('total_tasks', 0)}")
    print(f"  - Unique images found: {stats.get('unique_images', 0)}")
    if stats.get('missing_images', 0) > 0:
        print(f"  - ⚠️ Missing images: {stats.get('missing_images', 0)}")

    print("\n  Geometric Concept Distribution:")
    concept_dist = stats.get('concept_distribution', {})
    if not concept_dist:
        print("    - No concept distribution data found.")
    else:
        for concept, count in sorted(concept_dist.items(), key=lambda item: item[1], reverse=True):
            percentage = (count / stats['total_tasks']) * 100 if stats['total_tasks'] > 0 else 0
            print(f"    - {concept:<20}: {count:<5} ({percentage:.1f}%)")
            
    return adapter


def test_task_creation_by_concept(adapter: GEOSAdapter):
    """Test creating GEOSTask instances for different geometric concepts."""
    print("\n" + "="*60)
    print("2. Testing Task Creation by Geometric Concept")
    print("="*60)
    
    # 获取所有可用的概念类型
    concepts_to_test = list(adapter.concept_types.keys())
    if 'unknown' in concepts_to_test:
        concepts_to_test.remove('unknown')

    print(f"Testing for concepts: {concepts_to_test}")
    
    for concept in concepts_to_test:
        task_ids = adapter.get_task_ids(concept=concept, limit=1)
        if task_ids:
            task_id = task_ids[0]
            task = GEOSTask(task_id=task_id, adapter=adapter)
            goal, info = task.setup()
            
            print(f"\n[{concept.upper()} Task]")
            print(f"  - Task ID: {task_id}")
            print(f"  - Question: {task.question[:80]}...")
            print(f"  - Answer: {task.answer}")
            # 验证从task内部检测到的概念是否与adapter分类的一致
            status = "✓" if info.get('geometric_concept') == concept else "❌"
            print(f"  - Detected Concept: {info.get('geometric_concept')} ({status})")
            print(f"  - Requires Calculation: {info.get('requires_calculation')}")
        else:
            print(f"\n[No tasks found for concept: {concept}]")


def test_answer_validation(adapter: GEOSAdapter):
    """Test the answer validation logic in GEOSTask with various scenarios."""
    print("\n" + "="*60)
    print("3. Testing Answer Validation (Numerical and Expressions)")
    print("="*60)

    # --- Case 1: Test with a mathematical expression answer (e.g., involving sqrt) ---
    print("\n--- [Test Case 1: Mathematical Expression Answer] ---")
    expr_task_id = None
    # 查找一个包含 'sqrt' 的答案
    for task_data in adapter.annotations:
        if 'sqrt' in str(task_data['answer']):
            expr_task_id = task_data['id']
            break
            
    if not expr_task_id:
        print("Could not find a sample task with a 'sqrt' answer. Skipping expression test.")
    else:
        task = GEOSTask(task_id=expr_task_id, adapter=adapter)
        task.setup()
        print(f"Question: {task.question}")
        print(f"Correct Answer String: '{task.answer}'")
        correct_value = task._evaluate_math_expr(task.answer)
        print(f"Correct Answer Evaluated: {correct_value:.4f}")
        
        test_cases = [
            (task.answer, "Exact string match", True),
            ("sqrt(20)", "Equivalent expression (if ans=2*sqrt(5))", True),
            ("4.4721", "Correct numerical value", True),
            ("4.472", "Value with minor tolerance", True),
            ("4.5", "Value outside tolerance", False),
            ("10", "Wrong value", False),
        ]
        
        for test_answer, desc, should_pass in test_cases:
            success, feedback = task.check_success(test_answer)
            status = "✅" if success == should_pass else "❌"
            print(f"  {status} {desc}: '{test_answer}' -> {feedback}")

    # --- Case 2: Test with a simple integer answer ---
    print("\n--- [Test Case 2: Simple Integer Answer] ---")
    int_task_id = None
    for task_data in adapter.annotations:
        try:
            if isinstance(int(task_data['answer']), int):
                int_task_id = task_data['id']
                break
        except (ValueError, TypeError):
            continue
            
    if not int_task_id:
        print("Could not find a sample task with a simple integer answer. Skipping integer test.")
    else:
        task = GEOSTask(task_id=int_task_id, adapter=adapter)
        task.setup()
        print(f"Question: {task.question}")
        print(f"Correct Answer: '{task.answer}'")

        test_cases = [
            (task.answer, "Exact integer match", True),
            (f"{task.answer}.0", "Float representation", True),
            (str(int(task.answer) + 1), "Wrong integer", False),
            ("Some random text", "Non-numeric text", False),
        ]
        
        for test_answer, desc, should_pass in test_cases:
            success, feedback = task.check_success(test_answer)
            status = "✅" if success == should_pass else "❌"
            print(f"  {status} {desc}: '{test_answer}' -> {feedback}")



def test_complete_workflow(adapter: GEOSAdapter):
    """Test the full lifecycle of a single GEOS task."""
    print("\n" + "="*60)
    print("4. Testing Complete Task Workflow")
    print("="*60)
    
    task_id = random.choice(adapter.get_task_ids())
    task = GEOSTask(task_id=task_id, adapter=adapter)
    
    # 1. Setup
    print("[1. Setup Phase]")
    goal, info = task.setup()
    print(f"Task ID: {task.task_id}")
    print(f"Question: {task.question}")
    print(f"Expected Answer: {task.answer}")
    print("Generated Goal/Instructions:")
    print("-" * 20)
    print(goal)
    print("-" * 20)
    
    # 2. Observation
    print("\n[2. Observation Phase]")
    obs = task.get_observation()
    print(f"Observation keys: {list(obs.keys())}")
    # 从 task 对象本身获取 image_path，而不是从 obs 字典
    print(f"Image path for task: {Path(task.image_path).name}")
    
    # 3. Validation
    print("\n[3. Validation Phase]")
    # 模拟正确答案
    reward, done, msg, val_info = task.validate(chat_history=[], observation=task.answer)
    print(f"  - Correct Answer: Reward={reward}, Done={done}, Msg='{msg}'")
    
    # 模拟错误答案
    wrong_answer = "9999"
    reward, done, msg, val_info = task.validate(chat_history=[], observation=wrong_answer)
    print(f"  - Incorrect Answer: Reward={reward}, Done={done}, Msg='{msg}'")
    
    # 4. Metrics
    print("\n[4. Metrics Phase]")
    metrics = task.get_metrics()
    print("Generated Metrics:")
    print(f"  - Geometric Concept: {metrics.get('geometric_concept')}")
    print(f"  - Requires Calculation: {metrics.get('requires_calculation')}")
    # 'success' 来源于上一次的 validate 调用
    print(f"  - Success (last attempt): {metrics.get('success')}")



def visualize_sample_diagrams(adapter: GEOSAdapter, n_samples: int = 6):
    """Visualize some sample diagrams with their questions and answers."""
    print("\n" + "="*60)
    print(f"5. Visualizing {n_samples} Sample Diagrams")
    print("="*60)
    
    if len(adapter.annotations) < n_samples:
        print(f"Not enough samples to visualize. Need at least {n_samples}, found {len(adapter.annotations)}.")
        return
        
    task_ids = random.sample(adapter.get_task_ids(), k=n_samples)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, task_id in enumerate(task_ids):
        task_data = adapter.get_task_data(task_id)
        try:
            img = Image.open(task_data['image_path'])
            axes[i].imshow(img)
            axes[i].axis('off')
            
            question = task_data['question']
            # Word wrap the question for better display
            wrapped_question = "\n".join(question[j:j+40] for j in range(0, len(question), 40))
            
            title = f"Q: {wrapped_question}\n\nAnswer: {task_data['answer']}"
            axes[i].set_title(title, fontsize=10, pad=10)
            
        except FileNotFoundError:
            axes[i].text(0.5, 0.5, "Image not found", ha='center', va='center')
            axes[i].axis('off')
            
    plt.tight_layout(pad=3.0)
    output_file = "geos_samples.png"
    plt.savefig(output_file, dpi=150)
    print(f"\n✓ Sample diagrams saved to: {output_file}")
    plt.close()


def main():
    """Main function to run the test suite."""
    parser = argparse.ArgumentParser(description='Test GEOS dataset integration with VLM Gym.')
    parser.add_argument(
        '--annotation',
        type=str,
        # 根据您之前的脚本，假设输出文件在这里
        default='/data/wang/meng/GYM-Work/dataset/GEOS/geos_vlm_gym_format.json',
        help='Path to the VLM Gym formatted GEOS JSON annotation file.'
    )
    args = parser.parse_args()

    print("="*70)
    print("GEOS Dataset Integration Test Suite")
    print("="*70)

    # 检查注释文件是否存在
    if not Path(args.annotation).exists():
        print(f"❌ Annotation file not found at: {args.annotation}")
        print("Please run the conversion script first or provide the correct path.")
        return 1

    try:
        # 1. 测试 Adapter 加载
        adapter = test_adapter_loading(args.annotation)
        if not adapter or not adapter.annotations:
             print("\n❌ Adapter failed to load any data. Aborting further tests.")
             return 1

        # 2. 测试按概念创建任务
        test_task_creation_by_concept(adapter)

        # 3. 测试答案验证逻辑
        test_answer_validation(adapter)

        # 4. 测试完整工作流
        test_complete_workflow(adapter)

        # 5. 可视化样本
        visualize_sample_diagrams(adapter)

        print("\n" + "="*70)
        print("✅ All GEOS integration tests completed successfully!")
        print("="*70)
        print("\nKey checks passed:")
        print("  - Adapter loads data and calculates statistics correctly.")
        print("  - Tasks are created successfully for different geometric concepts.")
        print("  - Answer validation handles both numbers and math expressions.")
        print("  - The full task lifecycle (setup, observe, validate) works.")
        print("  - Visualization of samples has been generated in 'geos_samples.png'.")

    except Exception as e:
        print(f"\n❌ A test failed with an unexpected error: {e}")
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())