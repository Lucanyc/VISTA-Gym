#!/usr/bin/env python3
"""
使用ScienceQA数据集测试DeepEyes工具 - 增强版
- 保存所有图片到指定目录
- 显示hints和lecture内容
- 为选项添加标签
- 支持命令行参数控制数据量
"""

import sys
import os
import json
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import textwrap

# 添加项目路径
sys.path.append('/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista')

from vlm_gym.environments.tools.deepeyes_tool import DeepEyesTool

# 设置保存路径
SAVE_DIR = "/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/vlm_gym/environments/tools/crop_image/"

# 数据集路径
DATASET_JSON = "/data/wang/meng/GYM-Work/dataset/ScienceQA/reformatted/train.json"
DATASET_BASE = "/data/wang/meng/GYM-Work/dataset/ScienceQA"


def ensure_save_dir():
    """确保保存目录存在"""
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"Created save directory: {SAVE_DIR}")
    else:
        print(f"Using existing save directory: {SAVE_DIR}")


def load_scienceqa_dataset():
    """加载整个ScienceQA数据集"""
    with open(DATASET_JSON, 'r') as f:
        data = json.load(f)
    return data


def format_choices(choices):
    """为选项添加A、B、C、D标签"""
    formatted_choices = []
    for i, choice in enumerate(choices):
        label = chr(65 + i)  # A, B, C, D...
        formatted_choices.append(f"{label}. {choice}")
    return formatted_choices


def print_sample_info(sample_id, sample):
    """打印样本的详细信息，包括hints和lecture"""
    print(f"\n{'='*80}")
    print(f"Sample ID: {sample_id}")
    print(f"{'='*80}")
    
    # 基本信息
    print(f"\nQuestion: {sample['question']}")
    
    # 格式化选项
    formatted_choices = format_choices(sample['choices'])
    print(f"\nChoices:")
    for choice in formatted_choices:
        print(f"  {choice}")
    
    print(f"\nAnswer: {sample['answer']}")
    
    # 元数据
    metadata = sample.get('metadata', {})
    print(f"\nMetadata:")
    print(f"  Grade: {metadata.get('grade', 'N/A')}")
    print(f"  Subject: {metadata.get('original_subject', 'N/A')}")
    print(f"  Skills: {metadata.get('skills', ['N/A'])}")
    
    # Hint信息
    if 'hint' in sample and sample['hint']:
        print(f"\nHint:")
        wrapped_hint = textwrap.wrap(sample['hint'], width=76)
        for line in wrapped_hint:
            print(f"  {line}")
    
    # Lecture信息
    if 'lecture' in sample and sample['lecture']:
        print(f"\nLecture:")
        wrapped_lecture = textwrap.wrap(sample['lecture'], width=76)
        for line in wrapped_lecture[:5]:  # 只显示前5行
            print(f"  {line}")
        if len(wrapped_lecture) > 5:
            print(f"  ... (and {len(wrapped_lecture) - 5} more lines)")
    
    # 图像信息
    if sample.get('image'):
        print(f"\nImage: {sample['image']}")
        print(f"  Size: {metadata.get('img_width', 'N/A')}x{metadata.get('img_height', 'N/A')}")
    
    print(f"\n{'='*80}")


def save_image(image, filename, subfolder=None):
    """保存图像到指定目录"""
    if subfolder:
        save_path = os.path.join(SAVE_DIR, subfolder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        save_path = SAVE_DIR
    
    full_path = os.path.join(save_path, filename)
    image.save(full_path)
    print(f"Saved: {full_path}")
    return full_path


def test_single_sample(tool, sample_id, sample, timestamp):
    """测试单个样本"""
    print(f"\n--- Testing Sample {sample_id} ---")
    
    # 检查是否有图像
    if not sample.get('image'):
        print(f"Sample {sample_id} has no image, skipping...")
        return None
    
    # 加载图像
    image_path = os.path.join(DATASET_BASE, sample['image'])
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}, skipping...")
        return None
    
    # 显示样本信息
    print_sample_info(sample_id, sample)
    
    # 加载并保存原始图像
    original_image = Image.open(image_path)
    tool.reset(original_image)
    save_image(original_image, f"sample_{sample_id}_original_{timestamp}.png", f"sample_{sample_id}")
    
    results = {}
    
    # 测试1：缩放到上半部分（对于地图类问题很有用）
    print("\nTest 1: Zoom to upper half")
    upper_bbox = [0, 0, tool.width, tool.height // 2]
    zoom_upper_action = f"""
    <tool_call>
    {{"name": "image_zoom_in_tool", "arguments": {{"bbox_2d": {upper_bbox}}}}}
    </tool_call>
    """
    
    result_upper = tool.execute(zoom_upper_action)
    if result_upper.get('success'):
        save_image(result_upper['image'], f"sample_{sample_id}_zoom_upper_{timestamp}.png", f"sample_{sample_id}")
        results['zoom_upper'] = result_upper
        print("✓ Upper zoom successful")
    
    # 测试2：根据问题类型进行特定缩放
    tool.reset(original_image)
    
    # 如果是关于特定区域的问题，尝试缩放到中心
    if any(keyword in sample['question'].lower() for keyword in ['center', 'middle', 'focus']):
        print("\nTest 2: Zoom to center")
        center_bbox = [
            tool.width // 4,
            tool.height // 4,
            3 * tool.width // 4,
            3 * tool.height // 4
        ]
        zoom_center_action = f"""
        <tool_call>
        {{"name": "image_zoom_in_tool", "arguments": {{"bbox_2d": {center_bbox}}}}}
        </tool_call>
        """
        
        result_center = tool.execute(zoom_center_action)
        if result_center.get('success'):
            save_image(result_center['image'], f"sample_{sample_id}_zoom_center_{timestamp}.png", f"sample_{sample_id}")
            results['zoom_center'] = result_center
            print("✓ Center zoom successful")
    
    # 测试3：如果是实验类问题，可能需要旋转图像
    if 'experiment' in sample['question'].lower() or 'graph' in sample['question'].lower():
        tool.reset(original_image)
        print("\nTest 3: Rotate for better view")
        
        rotate_action = """
        <tool_call>
        {"name": "image_rotate_tool", "arguments": {"angle": 45}}
        </tool_call>
        """
        
        result_rotate = tool.execute(rotate_action)
        if result_rotate.get('success'):
            save_image(result_rotate['image'], f"sample_{sample_id}_rotate_45_{timestamp}.png", f"sample_{sample_id}")
            results['rotate'] = result_rotate
            print("✓ Rotation successful")
    
    # 测试4：模拟完整的推理过程
    tool.reset(original_image)
    print("\nTest 4: Full reasoning process")
    
    # 首先缩放
    reasoning_bbox = [0, 0, tool.width // 2, tool.height // 2]
    zoom_action = f"""
    <tool_call>
    {{"name": "image_zoom_in_tool", "arguments": {{"bbox_2d": {reasoning_bbox}}}}}
    </tool_call>
    """
    
    zoom_result = tool.execute(zoom_action)
    if zoom_result.get('success'):
        save_image(zoom_result['image'], f"sample_{sample_id}_reasoning_zoom_{timestamp}.png", f"sample_{sample_id}")
        
        # 然后给出答案
        # 创建一个带有hint和lecture上下文的推理
        answer_action = f"""
        Based on the question: "{sample['question']}"
        
        With the provided hint: "{sample.get('hint', 'No hint provided')[:100]}..."
        
        And considering the lecture content about {sample.get('metadata', {}).get('original_topic', 'the topic')},
        
        After examining the image carefully, the answer is:
        <answer>{sample['answer']}</answer>
        """
        
        answer_result = tool.execute(answer_action)
        if answer_result.get('final_answer'):
            print(f"✓ Answer extracted: {answer_result['final_answer']}")
            results['answer'] = answer_result
    
    return results


def create_comparison_visualization(sample_id, sample, timestamp):
    """创建对比可视化图"""
    image_path = os.path.join(DATASET_BASE, sample['image'])
    if not os.path.exists(image_path):
        return
    
    original_image = Image.open(image_path)
    
    # 创建一个展示原图、问题、选项、hints的综合图
    fig = plt.figure(figsize=(16, 10))
    
    # 左侧显示图像
    ax1 = plt.subplot(121)
    ax1.imshow(original_image)
    ax1.set_title(f"Sample {sample_id}: Original Image", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 右侧显示文本信息
    ax2 = plt.subplot(122)
    ax2.axis('off')
    
    # 准备文本内容
    text_content = f"Question:\n{textwrap.fill(sample['question'], width=50)}\n\n"
    
    text_content += "Choices:\n"
    for i, choice in enumerate(sample['choices']):
        label = chr(65 + i)
        wrapped_choice = textwrap.fill(f"{label}. {choice}", width=50, initial_indent='', subsequent_indent='   ')
        text_content += wrapped_choice + "\n"
    
    text_content += f"\nAnswer: {sample['answer']}\n\n"
    
    if sample.get('hint'):
        text_content += "Hint:\n"
        wrapped_hint = textwrap.fill(sample['hint'], width=50)
        text_content += wrapped_hint[:200] + "...\n\n" if len(wrapped_hint) > 200 else wrapped_hint + "\n\n"
    
    # 显示文本
    ax2.text(0.05, 0.95, text_content, transform=ax2.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, f"sample_{sample_id}_overview_{timestamp}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created overview visualization: {save_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Test DeepEyes tool with ScienceQA dataset')
    parser.add_argument('--num-samples', type=int, default=5, 
                        help='Number of samples to process (default: 5)')
    parser.add_argument('--start-idx', type=int, default=0,
                        help='Starting index for samples (default: 0)')
    parser.add_argument('--sample-ids', type=str, nargs='+',
                        help='Specific sample IDs to process (e.g., --sample-ids 1 3 5)')
    
    args = parser.parse_args()
    
    print("=== DeepEyes ScienceQA Enhanced Test ===")
    print(f"Dataset: {DATASET_JSON}")
    print(f"Save directory: {SAVE_DIR}")
    
    # 确保保存目录存在
    ensure_save_dir()
    
    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 加载数据集
    print("\nLoading dataset...")
    dataset = load_scienceqa_dataset()
    print(f"Total samples in dataset: {len(dataset)}")
    
    # 初始化工具
    tool = DeepEyesTool()
    
    # 确定要处理的样本
    if args.sample_ids:
        # 使用指定的样本ID
        sample_ids = args.sample_ids
        print(f"\nProcessing specific samples: {sample_ids}")
    else:
        # 使用索引范围
        all_ids = list(dataset.keys())
        end_idx = min(args.start_idx + args.num_samples, len(all_ids))
        sample_ids = all_ids[args.start_idx:end_idx]
        print(f"\nProcessing samples from index {args.start_idx} to {end_idx-1}")
    
    # 处理每个样本
    results_summary = {}
    for sample_id in sample_ids:
        if sample_id not in dataset:
            print(f"\nWarning: Sample ID {sample_id} not found in dataset")
            continue
        
        sample = dataset[sample_id]
        
        # 测试样本
        results = test_single_sample(tool, sample_id, sample, timestamp)
        if results:
            results_summary[sample_id] = results
            
            # 创建综合可视化
            create_comparison_visualization(sample_id, sample, timestamp)
    
    # 生成汇总报告
    summary_path = os.path.join(SAVE_DIR, f"test_summary_{timestamp}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"DeepEyes ScienceQA Enhanced Test Summary\n")
        f.write(f"{'='*80}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Dataset: {DATASET_JSON}\n")
        f.write(f"Save directory: {SAVE_DIR}\n")
        f.write(f"Samples processed: {len(results_summary)}\n\n")
        
        f.write("Processed samples:\n")
        for sample_id in results_summary:
            sample = dataset[sample_id]
            f.write(f"\n- Sample {sample_id}:\n")
            f.write(f"  Question: {sample['question'][:60]}...\n")
            f.write(f"  Grade: {sample.get('metadata', {}).get('grade', 'N/A')}\n")
            f.write(f"  Has hint: {'Yes' if sample.get('hint') else 'No'}\n")
            f.write(f"  Has lecture: {'Yes' if sample.get('lecture') else 'No'}\n")
            f.write(f"  Tests performed: {list(results_summary[sample_id].keys())}\n")
    
    print(f"\n{'='*80}")
    print("Test completed!")
    print(f"Summary saved to: {summary_path}")
    print(f"All results saved in: {SAVE_DIR}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()