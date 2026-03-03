# diagnose_medsam2_issues.py
"""
诊断MedSAM2在VQA-RAD数据集上的分割问题
"""
import sys
sys.path.append('/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista')

import json
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from vlm_gym.environments.tools.medsam2 import MedSAM2Tool
import cv2

# 路径配置
DATA_PATH = "/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/VQA-RAD/vqa_rad_train_vlmgym.json"
OUTPUT_DIR = "/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/vlm_gym/environments/tools/medsam2_diagnosis"

print("=== Diagnosing MedSAM2 Issues ===\n")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载数据
with open(DATA_PATH, 'r') as f:
    data = json.load(f)

# 初始化工具
tool = MedSAM2Tool({'device': 'cuda'})

# 测试不同类型的图像
test_cases = [
    # 原始的3个样本
    {"idx": 0, "description": "Brain MRI - Infarction"},
    {"idx": 1, "description": "Chest X-ray - Lungs"},
    {"idx": 2, "description": "Chest X-ray - Cardiovascular"},
    # 额外测试一些其他样本
    {"idx": 10, "description": "Sample 10"},
    {"idx": 20, "description": "Sample 20"},
]

print("1. Testing different prompt strategies on various images...\n")

for test_case in test_cases:
    idx = test_case["idx"]
    if idx >= len(data):
        continue
        
    sample = data[idx]
    print(f"\n{'='*60}")
    print(f"Testing {test_case['description']} (ID: {sample['id']})")
    print(f"Question: {sample['question']}")
    
    img_path = sample['image_path']
    if not os.path.exists(img_path):
        continue
        
    img = Image.open(img_path)
    img_np = np.array(img)
    
    # 分析图像特征
    print(f"\nImage analysis:")
    print(f"  - Size: {img.size}")
    print(f"  - Mode: {img.mode}")
    print(f"  - Pixel range: [{img_np.min()}, {img_np.max()}]")
    print(f"  - Mean intensity: {img_np.mean():.2f}")
    print(f"  - Std deviation: {img_np.std():.2f}")
    
    # 计算图像的边缘强度
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / edges.size
    print(f"  - Edge ratio: {edge_ratio:.3f}")
    
    # Reset工具
    tool.reset(img)
    
    # 测试多种提示策略
    strategies = [
        {
            "name": "Dense Grid (5x5)",
            "type": "point",
            "prompts": [
                [img.width*i/6, img.height*j/6, 1]
                for i in range(1, 6) for j in range(1, 6)
            ]
        },
        {
            "name": "Large Central Box",
            "type": "box",
            "prompts": [[img.width*0.1, img.height*0.1, img.width*0.9, img.height*0.9]]
        },
        {
            "name": "Mixed (Box + Points)",
            "type": "point",  # 实际上我们会分两次调用
            "prompts": "special"
        }
    ]
    
    results_summary = []
    
    for strategy in strategies:
        print(f"\n  Testing {strategy['name']}...")
        
        if strategy['prompts'] == "special":
            # 先用box
            result1 = tool.execute({
                "task": "segment",
                "prompt_type": "box",
                "prompts": [[img.width*0.3, img.height*0.3, img.width*0.7, img.height*0.7]],
                "frame_idx": 0,
                "propagate": False
            })
            # 再用points
            result2 = tool.execute({
                "task": "segment",
                "prompt_type": "point",
                "prompts": [[img.width/2, img.height/2, 1]],
                "frame_idx": 0,
                "propagate": False
            })
            
            # 合并结果
            success = result1.get('success') or result2.get('success')
            if success:
                area1 = result1.get('masks', {}).get('0', {}).get('area', 0) if result1.get('success') else 0
                area2 = result2.get('masks', {}).get('0', {}).get('area', 0) if result2.get('success') else 0
                area = max(area1, area2)
                print(f"    Mixed result - Area: {area}")
            else:
                area = 0
                print(f"    Both failed")
        else:
            result = tool.execute({
                "task": "segment",
                "prompt_type": strategy['type'],
                "prompts": strategy['prompts'],
                "frame_idx": 0,
                "propagate": False
            })
            
            success = result.get('success')
            area = result.get('masks', {}).get('0', {}).get('area', 0) if success else 0
            
        results_summary.append({
            'strategy': strategy['name'],
            'success': success,
            'area': area,
            'percentage': (area / (img.width * img.height) * 100) if area > 0 else 0
        })
        
        print(f"    Success: {success}, Area: {area} pixels ({results_summary[-1]['percentage']:.1f}%)")
    
    # 创建诊断图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # 原始图像
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # 直方图
    axes[1].hist(img_np.flatten(), bins=50, alpha=0.7)
    axes[1].set_title("Intensity Histogram")
    axes[1].set_xlabel("Pixel Value")
    axes[1].set_ylabel("Count")
    
    # 边缘检测
    axes[2].imshow(edges, cmap='gray')
    axes[2].set_title(f"Edge Detection\n(ratio: {edge_ratio:.3f})")
    axes[2].axis('off')
    
    # 结果摘要
    axes[3].axis('off')
    summary_text = f"Image: {sample['id']}\n"
    summary_text += f"Size: {img.size}\n"
    summary_text += f"Mean intensity: {img_np.mean():.1f}\n\n"
    summary_text += "Segmentation Results:\n"
    for res in results_summary:
        status = "✓" if res['success'] else "✗"
        summary_text += f"{status} {res['strategy']}: {res['area']} px ({res['percentage']:.1f}%)\n"
    axes[3].text(0.1, 0.9, summary_text, transform=axes[3].transAxes, 
                 verticalalignment='top', fontsize=10, family='monospace')
    
    # 最佳结果可视化（如果有）
    best_result = max(results_summary, key=lambda x: x['area'])
    if best_result['area'] > 100:
        axes[4].imshow(img)
        axes[4].set_title(f"Best Strategy: {best_result['strategy']}")
        axes[4].axis('off')
    else:
        axes[4].text(0.5, 0.5, "No meaningful\nsegmentation", 
                    transform=axes[4].transAxes, ha='center', va='center',
                    fontsize=14, color='red')
        axes[4].axis('off')
    
    # 问题和答案
    axes[5].axis('off')
    qa_text = f"Q: {sample['question']}\nA: {sample['answer']}"
    axes[5].text(0.1, 0.5, qa_text, transform=axes[5].transAxes,
                verticalalignment='center', fontsize=12, wrap=True)
    
    plt.tight_layout()
    
    # 保存
    output_path = os.path.join(OUTPUT_DIR, f"diagnosis_{sample['id']}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved diagnosis to: {output_path}")

# 2. 测试合成图像
print(f"\n{'='*60}")
print("\n2. Testing on synthetic images to verify tool functionality...")

# 创建一个高对比度的合成图像
synthetic_img = np.zeros((512, 512, 3), dtype=np.uint8)
# 添加一个明显的白色圆形
cv2.circle(synthetic_img, (256, 256), 150, (255, 255, 255), -1)
synthetic_pil = Image.fromarray(synthetic_img)

print("\nTesting synthetic circle image...")
tool.reset(synthetic_pil)

result = tool.execute({
    "task": "segment",
    "prompt_type": "point",
    "prompts": [[256, 256, 1]],  # 圆心
    "frame_idx": 0,
    "propagate": False
})

if result.get('success'):
    area = result.get('masks', {}).get('0', {}).get('area', 0)
    expected_area = np.pi * 150 * 150  # πr²
    accuracy = (area / expected_area) * 100 if expected_area > 0 else 0
    print(f"  ✓ Synthetic test successful!")
    print(f"  Segmented area: {area} pixels")
    print(f"  Expected area: {expected_area:.0f} pixels")
    print(f"  Accuracy: {accuracy:.1f}%")
else:
    print(f"  ✗ Even synthetic image failed!")
    print(f"  This suggests a fundamental issue with the tool/model")

print("\n=== Diagnosis Complete ===")
print(f"\nDiagnosis results saved to: {OUTPUT_DIR}")
print("\nConclusions:")
print("1. Check the synthetic image result - if it fails, the model has fundamental issues")
print("2. Compare edge ratios - low edge ratios might indicate poor contrast")
print("3. Look at intensity histograms - bimodal distributions might work better")