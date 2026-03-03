#!/usr/bin/env python3
"""
SAM2工具独立测试脚本 - 不依赖外部导入
"""

import sys
sys.path.append('/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista')
sys.path.append('/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/vlm_gym/environments/tools/segment-anything-2')

import os
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, Any, List, Tuple, Optional, Union

# 简化版ToolBase（如果需要）
class SimpleToolBase:
    """简化的工具基类"""
    def __init__(self):
        pass
    
    def reset(self, image):
        raise NotImplementedError
    
    def execute(self, action_string):
        raise NotImplementedError

# SAM2工具类
class SAM2Tool(SimpleToolBase):
    """
    SAM2工具 - 基于提示的图像分割
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        
        # 工具属性
        self.name = "sam2"
        self.description = "基于提示的图像分割工具"
        
        # 配置
        self.config = config or {}
        self.model_id = self.config.get('model_id', 'facebook/sam2-hiera-large')
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.save_visualizations = self.config.get('save_visualizations', False)
        self.output_dir = self.config.get('output_dir', './sam2_outputs')
        
        # 延迟加载
        self.predictor = None
        self.current_image = None
        self.current_image_array = None
        
        print(f"[SAM2] 初始化: model={self.model_id}, device={self.device}")
    
    def _load_predictor(self):
        """延迟加载SAM2预测器"""
        if self.predictor is None:
            print(f"[SAM2] 加载模型...")
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            self.predictor = SAM2ImagePredictor.from_pretrained(self.model_id)
            print("[SAM2] 模型加载成功")
    
    def reset(self, image: Union[str, Image.Image]):
        """重置工具状态"""
        self._load_predictor()
        
        # 处理输入
        if isinstance(image, str):
            self.current_image = Image.open(image).convert('RGB')
        else:
            self.current_image = image.convert('RGB')
        
        self.current_image_array = np.array(self.current_image)
        
        # 设置图像
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            self.predictor.set_image(self.current_image_array)
        
        print(f"[SAM2] 图像设置成功: {self.current_image.size}")
    
    def execute(self, action_string: str) -> Dict[str, Any]:
        """执行分割任务"""
        if self.current_image is None:
            return {"error": "No image loaded"}
        
        # 解析参数
        if isinstance(action_string, str):
            try:
                params = json.loads(action_string)
            except:
                params = {"task": action_string}
        else:
            params = action_string
        
        task = params.get("task", "point_segment")
        
        try:
            if task == "point_segment":
                return self._point_segment(params)
            elif task == "box_segment":
                return self._box_segment(params)
            elif task == "multi_point_segment":
                return self._multi_point_segment(params)
            else:
                return {"error": f"Unknown task: {task}"}
        except Exception as e:
            return {"error": str(e)}
    
    def _point_segment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """点分割"""
        h, w = self.current_image_array.shape[:2]
        
        # 获取或生成点
        points = params.get('points', [[w//2, h//2]])
        labels = params.get('labels', [1])
        
        points = np.array(points)
        labels = np.array(labels)
        
        # 预测
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            masks, scores, logits = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True
            )
        
        # 处理结果
        results = []
        for i, (mask, score) in enumerate(zip(masks, scores)):
            pixel_count = np.sum(mask)
            coverage = pixel_count / (h * w) * 100
            results.append({
                "mask_id": i,
                "score": float(score),
                "coverage_percent": float(coverage)
            })
        
        # 可视化
        if self.save_visualizations:
            self._visualize_result(points, labels, masks[np.argmax(scores)], "point_segment")
        
        return {
            "success": True,
            "task": "point_segment",
            "num_masks": len(masks),
            "results": results
        }
    
    def _box_segment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """框分割"""
        h, w = self.current_image_array.shape[:2]
        
        # 获取或生成框
        box = params.get('box', [w//4, h//4, 3*w//4, 3*h//4])
        box = np.array(box)
        
        # 预测
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            masks, scores, logits = self.predictor.predict(
                box=box,
                multimask_output=False
            )
        
        # 处理结果
        pixel_count = np.sum(masks[0])
        coverage = pixel_count / (h * w) * 100
        
        return {
            "success": True,
            "task": "box_segment",
            "score": float(scores[0]),
            "coverage_percent": float(coverage),
            "box": box.tolist()
        }
    
    def _multi_point_segment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """多点分割"""
        h, w = self.current_image_array.shape[:2]
        
        # 获取点集
        points = params.get('points', None)
        labels = params.get('labels', None)
        
        if points is None:
            # 默认5点
            points = [
                [w//2, h//2],
                [w//4, h//4],
                [3*w//4, h//4],
                [w//4, 3*h//4],
                [3*w//4, 3*h//4]
            ]
            labels = [1, 1, 1, 0, 0]
        
        points = np.array(points)
        labels = np.array(labels)
        
        # 预测
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            masks, scores, logits = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True
            )
        
        # 处理结果
        results = []
        for i, (mask, score) in enumerate(zip(masks, scores)):
            pixel_count = np.sum(mask)
            coverage = pixel_count / (h * w) * 100
            results.append({
                "mask_id": i,
                "score": float(score),
                "coverage_percent": float(coverage)
            })
        
        return {
            "success": True,
            "task": "multi_point_segment",
            "num_masks": len(masks),
            "results": results
        }
    
    def _visualize_result(self, points, labels, mask, name):
        """简单的可视化"""
        plt.figure(figsize=(10, 10))
        plt.imshow(self.current_image_array)
        
        # 显示掩码
        mask_image = np.zeros_like(self.current_image_array)
        mask_image[:, :, 2] = mask * 255  # 蓝色掩码
        plt.imshow(mask_image, alpha=0.5)
        
        # 显示点
        pos_points = points[labels == 1]
        neg_points = points[labels == 0]
        
        if len(pos_points) > 0:
            plt.scatter(pos_points[:, 0], pos_points[:, 1], 
                       color='green', marker='*', s=200, edgecolor='white')
        if len(neg_points) > 0:
            plt.scatter(neg_points[:, 0], neg_points[:, 1], 
                       color='red', marker='*', s=200, edgecolor='white')
        
        plt.title(name)
        plt.axis('off')
        
        # 保存
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{name}.png")
        plt.savefig(output_path)
        plt.close()
        print(f"[SAM2] 可视化保存到: {output_path}")


# 测试函数
def test_sam2():
    """测试SAM2工具"""
    print("="*60)
    print("SAM2工具测试")
    print("="*60)
    
    # 1. 初始化
    print("\n[1] 初始化SAM2工具...")
    config = {
        'device': 'cuda',
        'save_visualizations': True,
        'output_dir': '/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/vlm_gym/environments/tools/sam2_test_outputs'
    }
    
    sam2 = SAM2Tool(config)
    
    # 2. 加载测试图像
    print("\n[2] 加载测试图像...")
    test_image = "/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/VQA-RAD/train/images/image_00000.png"
    
    try:
        sam2.reset(test_image)
        print("✓ 图像加载成功")
    except Exception as e:
        print(f"✗ 图像加载失败: {e}")
        return
    
    # 3. 测试点分割
    print("\n[3] 测试点分割...")
    result1 = sam2.execute(json.dumps({
        "task": "point_segment",
        "points": [[283, 277]],
        "labels": [1]
    }))
    
    if result1.get('success'):
        print("✓ 点分割成功")
        print(f"  掩码数: {result1['num_masks']}")
        for r in result1['results']:
            print(f"  掩码{r['mask_id']}: 分数={r['score']:.3f}, 覆盖率={r['coverage_percent']:.1f}%")
    
    # 4. 测试框分割
    print("\n[4] 测试框分割...")
    result2 = sam2.execute(json.dumps({
        "task": "box_segment",
        "box": [100, 100, 400, 400]
    }))
    
    if result2.get('success'):
        print("✓ 框分割成功")
        print(f"  分数: {result2['score']:.3f}")
        print(f"  覆盖率: {result2['coverage_percent']:.1f}%")
    
    # 5. 测试多点分割
    print("\n[5] 测试多点分割...")
    result3 = sam2.execute(json.dumps({
        "task": "multi_point_segment"
    }))
    
    if result3.get('success'):
        print("✓ 多点分割成功")
        print(f"  掩码数: {result3['num_masks']}")
        best = max(result3['results'], key=lambda x: x['score'])
        print(f"  最佳掩码: 分数={best['score']:.3f}, 覆盖率={best['coverage_percent']:.1f}%")
    
    print("\n" + "="*60)
    print("测试完成！")
    print(f"结果保存在: {config['output_dir']}")


if __name__ == "__main__":
    test_sam2()