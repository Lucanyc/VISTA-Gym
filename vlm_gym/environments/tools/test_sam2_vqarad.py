import sys
sys.path.append('/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista')

import json
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches

try:
    # 导入SAM2工具
    from vlm_gym.environments.tools import SAM2Tool
    print("SAM2Tool import successful")
    
    # 创建SAM2工具实例
    sam2_config = {
        'device': 'cuda',
        'save_visualizations': True,
        'output_dir': '/workspace/mathvista/sam2_vqarad_output'
    }
    
    sam2 = SAM2Tool(sam2_config)
    print("SAM2Tool created successfully")
    
    # 检查SAM2的能力
    capabilities = sam2.get_capabilities()
    print("\nSAM2 Capabilities:")
    print(f"- Name: {capabilities['name']}")
    print(f"- Description: {capabilities['description']}")
    print(f"- Available tasks: {capabilities['parameters']['task']['enum']}")
    
    # 加载VQA-RAD数据
    data_path = "/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/VQA-RAD/vqa_rad_train_vlmgym.json"
    
    print(f"\nLoading data from: {data_path}")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    # 创建输出目录
    output_dir = "/workspace/mathvista/sam2_vqarad_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # 测试前5个样本
    for i, sample in enumerate(data[:5]):
        print(f"\n{'='*60}")
        print(f"Sample {i+1}: {sample['id']}")
        print(f"Question: {sample['question']}")
        print(f"Answer: {sample['answer']}")
        print(f"Task: {sample.get('task', 'N/A')}")
        
        # 获取图片路径
        image_path = sample['image_path']
        print(f"\nImage path: {image_path}")
        print(f"Image exists: {os.path.exists(image_path)}")
        
        if os.path.exists(image_path):
            # 加载图像
            img = Image.open(image_path)
            print(f"Image size: {img.size}")
            print(f"Image mode: {img.mode}")
            
            # 重置SAM2工具
            print(f"\nResetting SAM2 with image...")
            sam2.reset(image_path)
            
            # 策略1: 智能医学分割（基于问题内容）
            print("\n[Strategy 1] Smart Medical Segmentation")
            smart_action = {
                "task": "smart_medical_segment",
                "question": sample['question'],
                "save_visualization": True
            }
            
            smart_result = sam2.execute(json.dumps(smart_action))
            
            if smart_result.get('success'):
                print(f"✓ Smart segmentation successful")
                print(f"  - Detected organ: {smart_result.get('detected_organ', 'None')}")
                print(f"  - Strategy: {smart_result.get('strategy', 'Default')}")
                print(f"  - Number of masks: {smart_result.get('num_masks', 0)}")
                
                if 'results' in smart_result:
                    # 显示每个掩码的信息
                    for mask_info in smart_result['results']:
                        print(f"  - Mask {mask_info['mask_id']}: "
                              f"Score={mask_info['score']:.3f}, "
                              f"Coverage={mask_info['coverage_percent']:.1f}%, "
                              f"BBox={mask_info['bbox']}")
            else:
                print(f"✗ Smart segmentation failed: {smart_result.get('error')}")
            
            # 策略2: 中心点分割（作为对比）
            print("\n[Strategy 2] Center Point Segmentation")
            center_action = {
                "task": "point_segment",
                "save_visualization": False
            }
            
            center_result = sam2.execute(json.dumps(center_action))
            
            if center_result.get('success'):
                print(f"✓ Center point segmentation successful")
                print(f"  - Number of masks: {center_result.get('num_masks', 0)}")
                
                if 'results' in center_result:
                    best_mask = max(center_result['results'], key=lambda x: x['score'])
                    print(f"  - Best mask: Score={best_mask['score']:.3f}, "
                          f"Coverage={best_mask['coverage_percent']:.1f}%")
            
            # 策略3: 多点分割
            print("\n[Strategy 3] Multi-point Segmentation")
            multi_action = {
                "task": "multi_point_segment",
                "save_visualization": False
            }
            
            multi_result = sam2.execute(json.dumps(multi_action))
            
            if multi_result.get('success'):
                print(f"✓ Multi-point segmentation successful")
                print(f"  - Number of masks: {multi_result.get('num_masks', 0)}")
                
                if 'results' in multi_result:
                    best_mask = max(multi_result['results'], key=lambda x: x['score'])
                    print(f"  - Best mask: Score={best_mask['score']:.3f}, "
                          f"Coverage={best_mask['coverage_percent']:.1f}%")
            
            # 策略4: 基于图像内容的框分割
            print("\n[Strategy 4] Adaptive Box Segmentation")
            # 根据问题内容调整框的位置
            h, w = img.size[1], img.size[0]
            
            if 'brain' in sample['question'].lower():
                # 脑部通常在上半部分
                box = [w*0.2, h*0.1, w*0.8, h*0.6]
            elif 'lung' in sample['question'].lower():
                # 肺部在中间区域
                box = [w*0.2, h*0.3, w*0.8, h*0.8]
            elif 'heart' in sample['question'].lower():
                # 心脏在中心偏左
                box = [w*0.3, h*0.35, w*0.6, h*0.65]
            else:
                # 默认中心区域
                box = [w*0.25, h*0.25, w*0.75, h*0.75]
            
            box_action = {
                "task": "box_segment",
                "box": [int(b) for b in box],
                "save_visualization": False
            }
            
            box_result = sam2.execute(json.dumps(box_action))
            
            if box_result.get('success'):
                print(f"✓ Box segmentation successful")
                print(f"  - Box coordinates: {box_action['box']}")
                if 'results' in box_result:
                    print(f"  - Coverage: {box_result['results'][0]['coverage_percent']:.1f}%")
                    print(f"  - Score: {box_result['results'][0]['score']:.3f}")
            
            # 创建综合可视化
            print("\n[Creating Combined Visualization]")
            try:
                fig, axes = plt.subplots(2, 2, figsize=(15, 15))
                
                # 原图
                axes[0, 0].imshow(img)
                axes[0, 0].set_title("Original Image")
                axes[0, 0].axis('off')
                
                # 智能分割结果（如果有）
                if smart_result.get('success') and 'results' in smart_result:
                    axes[0, 1].imshow(img)
                    axes[0, 1].set_title(f"Smart Segmentation\n(Organ: {smart_result.get('detected_organ', 'None')})")
                    axes[0, 1].axis('off')
                    # 这里可以添加掩码可视化（如果有掩码数据）
                
                # 中心点分割结果
                if center_result.get('success') and 'results' in center_result:
                    axes[1, 0].imshow(img)
                    best_mask = max(center_result['results'], key=lambda x: x['score'])
                    axes[1, 0].set_title(f"Center Point\n(Coverage: {best_mask['coverage_percent']:.1f}%)")
                    axes[1, 0].axis('off')
                
                # 框分割结果
                if box_result.get('success'):
                    axes[1, 1].imshow(img)
                    # 画框
                    rect = patches.Rectangle(
                        (box_action['box'][0], box_action['box'][1]),
                        box_action['box'][2] - box_action['box'][0],
                        box_action['box'][3] - box_action['box'][1],
                        linewidth=2, edgecolor='green', facecolor='none'
                    )
                    axes[1, 1].add_patch(rect)
                    axes[1, 1].set_title(f"Box Segmentation\n(Coverage: {box_result['results'][0]['coverage_percent']:.1f}%)")
                    axes[1, 1].axis('off')
                
                # 添加问题和答案
                fig.suptitle(f"Sample: {sample['id']}\nQ: {sample['question']}\nA: {sample['answer']}", 
                           fontsize=12, y=0.98)
                
                # 保存综合图
                output_path = os.path.join(output_dir, f"combined_{sample['id']}.png")
                plt.tight_layout()
                plt.savefig(output_path, bbox_inches='tight', dpi=150)
                plt.close()
                print(f"✓ Combined visualization saved to: {output_path}")
                
            except Exception as e:
                print(f"✗ Error creating visualization: {e}")
            
        else:
            print(f"✗ Image not found: {image_path}")
    
    # 统计分析
    print(f"\n{'='*60}")
    print("Dataset Analysis with SAM2:")
    
    # 分析前100个样本的器官分布
    organ_counts = {'brain': 0, 'lung': 0, 'heart': 0, 'other': 0}
    coverage_stats = []
    
    print("\nAnalyzing first 100 samples for organ distribution...")
    for i, sample in enumerate(data[:100]):
        if i % 20 == 0:
            print(f"  Processing sample {i}...")
        
        question_lower = sample['question'].lower()
        
        # 统计器官类型
        if any(word in question_lower for word in ['brain', 'cerebral', 'cranial', 'skull', 'head']):
            organ_counts['brain'] += 1
        elif any(word in question_lower for word in ['lung', 'pulmonary', 'chest', 'thorax']):
            organ_counts['lung'] += 1
        elif any(word in question_lower for word in ['heart', 'cardiac', 'cardiovascular']):
            organ_counts['heart'] += 1
        else:
            organ_counts['other'] += 1
    
    print("\nOrgan distribution in questions:")
    for organ, count in organ_counts.items():
        percentage = count / 100 * 100
        print(f"  - {organ.capitalize()}: {count} ({percentage:.1f}%)")
    
    # 分析任务类型
    task_counts = {}
    for sample in data[:100]:
        task = sample.get('task', 'unknown')
        task_counts[task] = task_counts.get(task, 0) + 1
    
    print("\nTask type distribution:")
    for task, count in sorted(task_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {task}: {count}")
    
    # 测试特定功能
    print(f"\n{'='*60}")
    print("Testing SAM2 special features:")
    
    # 找一个脑部图像测试
    brain_sample = next((s for s in data if 'brain' in s['question'].lower()), None)
    if brain_sample and os.path.exists(brain_sample['image_path']):
        print(f"\nTesting on brain image: {brain_sample['id']}")
        sam2.reset(brain_sample['image_path'])
        
        # 测试不同的multimask设置
        print("\nTesting multimask_output variations:")
        
        # 单掩码输出
        single_result = sam2.execute(json.dumps({
            "task": "point_segment",
            "multimask_output": False
        }))
        print(f"  - Single mask: {single_result.get('num_masks', 0)} masks")
        
        # 多掩码输出
        multi_result = sam2.execute(json.dumps({
            "task": "point_segment",
            "multimask_output": True
        }))
        print(f"  - Multi mask: {multi_result.get('num_masks', 0)} masks")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# 总结
print(f"\n{'='*60}")
print("Summary:")
print("1. SAM2可以对医学图像进行分割")
print("2. 对于VQA-RAD数据集，主要功能：")
print("   - 智能医学分割：根据问题内容自动选择分割策略")
print("   - 点分割：使用指定点或默认中心点")
print("   - 框分割：使用边界框定义感兴趣区域")
print("   - 多点分割：结合前景点和背景点")
print("\n注意事项：")
print("- 医学图像的对比度和普通图像不同，可能需要调整提示策略")
print("- 不同器官需要不同的分割策略")
print("- SAM2是提示驱动的，需要提供合适的点或框")
print("- 可以结合问题内容智能选择分割区域")
print("\n建议：")
print("- 可以先用其他工具（如目标检测）定位器官，再用SAM2精细分割")
print("- 对于复杂问题，可以组合多种分割策略")
print("- 分割结果可以帮助VLM更好地理解医学图像")