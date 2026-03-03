# sam2_tool.py
"""
SAM2工具 - 与VLM Gym框架集成的版本
支持官方的三种prompt类型：Point, Box, Mask
"""

import sys
sys.path.append('/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista')

import os
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, Any, List, Tuple, Optional, Union

from vlm_gym.environments.tools.base import ToolBase


class SAM2Tool(ToolBase):
    """
    SAM2工具 - 支持官方三种prompt类型的图像分割
    
    Prompt类型：
    1. Point prompts: 点提示（正/负样本）
    2. Box prompts: 框提示
    3. Mask prompts: 遮罩提示
    """
    # 类级别属性
    name = "sam2"
    
    def __init__(self, config: Dict[str, Any] = None):
        # 先调用父类初始化，传递name参数
        super().__init__(name=self.name)
        
        # 设置工具描述和能力
        self.description = "SAM2图像分割工具，支持Point、Box、Mask三种官方prompt类型及其组合"
        self.capabilities = [
            "点提示分割（Point prompts）", 
            "框提示分割（Box prompts）", 
            "遮罩提示分割（Mask prompts）",
            "组合提示分割（Combined prompts）",
            "交互式精修（Refinement）",
            "智能医学分割（高级功能）"
        ]
        
        # 配置
        self.config = config or {}
        
        # 模型相关配置
        self.model_id = self.config.get('model_id', 'facebook/sam2-hiera-large')
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.use_bfloat16 = self.config.get('use_bfloat16', True)
        
        # 输出相关配置
        self.save_visualizations = self.config.get('save_visualizations', False)
        self.output_dir = self.config.get('output_dir', './sam2_outputs')
        
        # 延迟加载模型
        self.predictor = None
        self.current_image = None
        self.current_image_array = None
        
        # 存储上一次的分割结果，用于refinement
        self.last_masks = None
        self.last_scores = None
        self.last_logits = None
        
        # 医学关键词映射（保留作为高级功能）
        self.medical_keywords = {
            'brain': {
                'keywords': ['brain', 'cerebral', 'cranial', 'skull', 'head'],
                'typical_points': [(0.5, 0.3), (0.3, 0.3), (0.7, 0.3)],
                'typical_box': [0.2, 0.1, 0.8, 0.6]
            },
            'lung': {
                'keywords': ['lung', 'pulmonary', 'chest', 'thorax'],
                'typical_points': [(0.3, 0.5), (0.7, 0.5)],
                'typical_box': [0.2, 0.3, 0.8, 0.8]
            },
            'heart': {
                'keywords': ['heart', 'cardiac', 'cardiovascular'],
                'typical_points': [(0.5, 0.5)],
                'typical_box': [0.35, 0.35, 0.65, 0.65]
            }
        }
        
        print(f"[SAM2] Initialized with model: {self.model_id}")
        print(f"[SAM2] Device: {self.device}")
        print(f"[SAM2] Supported prompts: Point, Box, Mask (and combinations)")
    
    def _load_predictor(self):
        """延迟加载SAM2预测器"""
        if self.predictor is None:
            print(f"[SAM2] Loading model: {self.model_id}")
            try:
                import os
                import sys
                
                # 首先移除可能存在的MedSAM2路径
                medsam2_path = "/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/MedSAM2"
                if medsam2_path in sys.path:
                    sys.path.remove(medsam2_path)
                    print(f"[SAM2] Removed MedSAM2 from path")
                
                # 移除已加载的sam2模块（如果是从MedSAM2加载的）
                modules_to_remove = [key for key in sys.modules.keys() if key.startswith('sam2')]
                for module in modules_to_remove:
                    del sys.modules[module]
                    print(f"[SAM2] Removed module: {module}")
                
                # 使用标准SAM2
                sam2_path = "/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/vlm_gym/environments/tools/segment-anything-2"
                if sam2_path not in sys.path:
                    sys.path.insert(0, sam2_path)
                print(f"[SAM2] Using SAM2 from: {sam2_path}")
                
                # 保存原始工作目录
                original_cwd = os.getcwd()
                
                # 尝试加载模型权重
                checkpoint_path = None
                
                # 查找权重文件的可能位置
                tools_dir = "/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/vlm_gym/environments/tools"
                possible_checkpoint_paths = [
                    # 从你提供的路径
                    os.path.join(tools_dir, "sam2_models/sam2-hiera-large/sam2_hiera_large.pt"),
                    # Hugging Face下载的其他可能位置
                    os.path.join(tools_dir, "sam2_models/models--facebook--sam2-hiera-large/blobs/*/sam2_hiera_large.pt"),
                    # 标准checkpoints目录
                    os.path.join(sam2_path, "checkpoints/sam2_hiera_large.pt"),
                ]
                
                # 查找实际存在的权重文件
                for path_pattern in possible_checkpoint_paths:
                    if '*' in path_pattern:
                        # 处理通配符
                        import glob
                        matches = glob.glob(path_pattern)
                        if matches:
                            checkpoint_path = matches[0]
                            break
                    elif os.path.exists(path_pattern):
                        checkpoint_path = path_pattern
                        break
                
                if checkpoint_path:
                    print(f"[SAM2] Found checkpoint: {checkpoint_path}")
                else:
                    print(f"[SAM2] No checkpoint found, will load without weights")
                
                # 改变工作目录到SAM2（Hydra需要）
                os.chdir(sam2_path)
                
                try:
                    # 导入SAM2模块
                    from sam2.build_sam import build_sam2
                    from sam2.sam2_image_predictor import SAM2ImagePredictor
                    
                    # 设置Hydra
                    from hydra import initialize_config_dir, compose
                    from hydra.core.global_hydra import GlobalHydra
                    
                    # 清理之前的Hydra实例
                    GlobalHydra.instance().clear()
                    
                    # 初始化Hydra配置目录
                    config_dir = os.path.join(sam2_path, "sam2", "configs")
                    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
                        # 根据model_id选择配置文件
                        if "large" in self.model_id or "hiera-large" in self.model_id:
                            config_name = "sam2/sam2_hiera_l.yaml"
                        elif "base-plus" in self.model_id:
                            config_name = "sam2/sam2_hiera_b+.yaml"
                        elif "small" in self.model_id:
                            config_name = "sam2/sam2_hiera_s.yaml"
                        elif "tiny" in self.model_id:
                            config_name = "sam2/sam2_hiera_t.yaml"
                        else:
                            config_name = "sam2/sam2_hiera_l.yaml"
                        
                        print(f"[SAM2] Using config: {config_name}")
                        
                        # 构建模型
                        if checkpoint_path:
                            sam2_model = build_sam2(config_name, ckpt_path=checkpoint_path, device=self.device)
                        else:
                            print(f"[SAM2] Warning: Loading without pre-trained weights!")
                            sam2_model = build_sam2(config_name, device=self.device)
                        
                        self.predictor = SAM2ImagePredictor(sam2_model)
                        print(f"[SAM2] Model loaded successfully")
                        
                except Exception as e:
                    print(f"[SAM2] Error during model loading: {e}")
                    raise
                finally:
                    # 恢复原始工作目录
                    os.chdir(original_cwd)
                    
            except Exception as e:
                print(f"[SAM2] Error loading model: {e}")
                import traceback
                traceback.print_exc()
                raise
    
    def reset(self, image: Union[str, Image.Image, np.ndarray]):
        """重置工具状态，准备处理新图像"""
        self._load_predictor()
        
        # 处理输入
        if isinstance(image, str):
            self.current_image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            self.current_image = Image.fromarray(image).convert('RGB')
        else:
            self.current_image = image.convert('RGB')
        
        # 转换为numpy数组
        self.current_image_array = np.array(self.current_image)
        
        # 设置图像到预测器
        with torch.inference_mode():
            if self.use_bfloat16 and self.device == 'cuda':
                with torch.autocast(self.device, dtype=torch.bfloat16):
                    self.predictor.set_image(self.current_image_array)
            else:
                self.predictor.set_image(self.current_image_array)
        
        # 清除之前的结果
        self.last_masks = None
        self.last_scores = None
        self.last_logits = None
        
        print(f"[SAM2] Reset with image size: {self.current_image.size}")
    
    def execute(self, action_string: str) -> Dict[str, Any]:
        """
        执行分割任务
        
        支持的prompt类型：
        - point_prompts: 点提示 [[x1,y1], [x2,y2], ...]
        - point_labels: 点标签 [1, 0, ...] (1=正样本, 0=负样本)
        - box_prompt: 框提示 [x1, y1, x2, y2]
        - mask_prompt: 遮罩提示 (numpy array or list)
        """
        
        if self.current_image is None:
            return {"error": "No image loaded. Please call reset() first."}
        
        # 解析参数
        if isinstance(action_string, str):
            try:
                params = json.loads(action_string)
            except json.JSONDecodeError:
                # 尝试解析为简单任务名称
                params = {"task": action_string}
        else:
            params = action_string
        
        print(f"[SAM2] Executing with params: {params}")
        
        try:
            # 检查是否是特殊任务
            task = params.get("task", None)
            
            if task == "smart_medical_segment":
                # 保留智能医学分割作为高级功能
                return self._smart_medical_segment(params)
            elif task == "refinement":
                # 精修功能
                return self._refinement_segment(params)
            else:
                # 标准prompt分割
                return self._standard_segment(params)
                
        except Exception as e:
            import traceback
            return {
                "error": f"Segmentation failed: {str(e)}",
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()
            }
    
    def _standard_segment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        标准分割：支持三种官方prompt类型及其组合
        """
        # 获取任务类型（如果没有指定，默认为standard）
        task = params.get("task", "standard")
        
        # 收集所有prompts
        prompt_kwargs = {}
        prompt_types_used = []
        
        # 1. Point prompts
        if 'point_prompts' in params or 'points' in params:
            points = params.get('point_prompts', params.get('points', None))
            labels = params.get('point_labels', params.get('labels', None))
            
            if points is not None:
                points = np.array(points)
                labels = np.array(labels) if labels else np.ones(len(points))
                
                prompt_kwargs['point_coords'] = points
                prompt_kwargs['point_labels'] = labels
                prompt_types_used.append('point')
        
        # 2. Box prompt
        if 'box_prompt' in params or 'box' in params:
            box = params.get('box_prompt', params.get('box', None))
            
            if box is not None:
                box = np.array(box)
                prompt_kwargs['box'] = box
                prompt_types_used.append('box')
        
        # 3. Mask prompt
        if 'mask_prompt' in params or 'mask' in params:
            mask = params.get('mask_prompt', params.get('mask', None))
            
            if mask is not None:
                # 处理mask输入（可能是list或numpy array）
                if isinstance(mask, list):
                    mask = np.array(mask)
                
                # 确保mask是正确的形状
                if mask.ndim == 2:
                    # 添加batch维度
                    mask = mask[np.newaxis, :, :]
                
                prompt_kwargs['mask_input'] = mask
                prompt_types_used.append('mask')
        
        # 如果没有提供任何prompt，使用默认的中心点
        if not prompt_kwargs:
            h, w = self.current_image_array.shape[:2]
            prompt_kwargs['point_coords'] = np.array([[w//2, h//2]])
            prompt_kwargs['point_labels'] = np.array([1])
            prompt_types_used.append('point')
        
        # 设置multimask_output
        multimask_output = params.get('multimask_output', True)
        
        # 执行预测
        masks, scores, logits = self._predict(
            **prompt_kwargs,
            multimask_output=multimask_output
        )
        
        # 保存结果用于后续refinement
        self.last_masks = masks
        self.last_scores = scores
        self.last_logits = logits
        
        # 处理结果
        results = self._process_masks(masks, scores)
        
        # 可视化
        if self.save_visualizations or params.get('save_visualization', False):
            self._save_advanced_visualization(
                prompt_kwargs, 
                masks[np.argmax(scores)], 
                f"segment_{'_'.join(prompt_types_used)}"
            )
        
        return {
            "success": True,
            "task": task,  # 添加任务类型
            "prompt_types": prompt_types_used,
            "num_masks": len(masks),
            "results": results,
            "best_mask_idx": int(np.argmax(scores)),
            "prompts_used": {
                "points": prompt_kwargs.get('point_coords', []).tolist() if 'point_coords' in prompt_kwargs else None,
                "labels": prompt_kwargs.get('point_labels', []).tolist() if 'point_labels' in prompt_kwargs else None,
                "box": prompt_kwargs.get('box', []).tolist() if 'box' in prompt_kwargs else None,
                "has_mask": 'mask_input' in prompt_kwargs
            }
        }
    
    def _refinement_segment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        精修分割：基于之前的结果进行改进
        """
        if self.last_masks is None:
            return {"error": "No previous segmentation to refine. Please perform initial segmentation first."}
        
        # 获取要精修的mask索引
        mask_idx = params.get('mask_idx', np.argmax(self.last_scores))
        
        # 使用之前的mask作为mask prompt
        previous_mask = self.last_masks[mask_idx]
        
        # 收集新的prompts用于精修
        new_params = params.copy()
        new_params['mask_prompt'] = previous_mask
        new_params['task'] = 'refinement'  # 设置任务类型
        
        # 执行新的分割
        return self._standard_segment(new_params)
    
    def _smart_medical_segment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """智能医学图像分割（高级功能）"""
        question = params.get('question', '').lower()
        
        # 识别医学关键词
        detected_organ = None
        for organ, info in self.medical_keywords.items():
            if any(keyword in question for keyword in info['keywords']):
                detected_organ = organ
                break
        
        if detected_organ:
            # 使用器官特定的提示策略
            h, w = self.current_image_array.shape[:2]
            organ_info = self.medical_keywords[detected_organ]
            
            # 构建组合prompt
            new_params = {
                'point_prompts': [[int(w * x), int(h * y)] for x, y in organ_info['typical_points']],
                'point_labels': [1] * len(organ_info['typical_points']),
                'box_prompt': [
                    int(w * organ_info['typical_box'][0]),
                    int(h * organ_info['typical_box'][1]),
                    int(w * organ_info['typical_box'][2]),
                    int(h * organ_info['typical_box'][3])
                ],
                'multimask_output': True,
                'task': 'smart_medical_segment'  # 设置任务类型
            }
            
            # 执行标准分割
            result = self._standard_segment(new_params)
            result['detected_organ'] = detected_organ
            result['strategy'] = f"Used {detected_organ}-specific prompts"
            
            return result
        else:
            # 回退到默认多点策略
            h, w = self.current_image_array.shape[:2]
            new_params = {
                'point_prompts': [
                    [w//2, h//2],      # 中心
                    [w//4, h//4],      # 左上
                    [3*w//4, h//4],    # 右上
                    [w//4, 3*h//4],    # 左下
                    [3*w//4, 3*h//4]   # 右下
                ],
                'point_labels': [1, 1, 1, 0, 0],  # 前3个前景，后2个背景
                'multimask_output': True,
                'task': 'smart_medical_segment'  # 设置任务类型
            }
            
            result = self._standard_segment(new_params)
            result['strategy'] = "Used default multi-point strategy"
            
            return result
    
    def _predict(self, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """执行预测"""
        with torch.inference_mode():
            if self.use_bfloat16 and self.device == 'cuda':
                with torch.autocast(self.device, dtype=torch.bfloat16):
                    return self.predictor.predict(**kwargs)
            else:
                return self.predictor.predict(**kwargs)
    
    def _process_masks(self, masks: np.ndarray, scores: np.ndarray) -> List[Dict[str, Any]]:
        """处理分割掩码"""
        h, w = self.current_image_array.shape[:2]
        results = []
        
        for i, (mask, score) in enumerate(zip(masks, scores)):
            # 计算掩码统计信息
            pixel_count = np.sum(mask)
            coverage = pixel_count / (h * w) * 100
            
            # 找到边界框
            if pixel_count > 0:
                y_indices, x_indices = np.where(mask)
                bbox = [
                    int(x_indices.min()),
                    int(y_indices.min()),
                    int(x_indices.max()),
                    int(y_indices.max())
                ]
                
                # 计算质心
                centroid_x = int(np.mean(x_indices))
                centroid_y = int(np.mean(y_indices))
                centroid = [centroid_x, centroid_y]
            else:
                bbox = [0, 0, 0, 0]
                centroid = [0, 0]
            
            results.append({
                "mask_id": i,
                "score": float(score),
                "pixel_count": int(pixel_count),
                "coverage_percent": float(coverage),
                "bbox": bbox,
                "centroid": centroid,
                "is_empty": pixel_count == 0
            })
        
        return results
    
    def _save_advanced_visualization(self, prompts, mask, name):
        """保存高级可视化结果"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(self.current_image_array)
        
        # 显示掩码
        h, w = mask.shape[-2:]
        color = np.array([30/255, 144/255, 255/255, 0.6])
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
        
        # 显示点提示
        if 'point_coords' in prompts:
            points = prompts['point_coords']
            labels = prompts['point_labels']
            
            pos_points = points[labels == 1]
            neg_points = points[labels == 0]
            
            if len(pos_points) > 0:
                ax.scatter(pos_points[:, 0], pos_points[:, 1], 
                          color='green', marker='*', s=375, 
                          edgecolor='white', linewidth=1.25,
                          label='Positive points')
            
            if len(neg_points) > 0:
                ax.scatter(neg_points[:, 0], neg_points[:, 1], 
                          color='red', marker='*', s=375, 
                          edgecolor='white', linewidth=1.25,
                          label='Negative points')
        
        # 显示框提示
        if 'box' in prompts:
            box = prompts['box']
            rect = patches.Rectangle(
                (box[0], box[1]), 
                box[2] - box[0], 
                box[3] - box[1],
                linewidth=2, 
                edgecolor='yellow', 
                facecolor='none',
                label='Box prompt'
            )
            ax.add_patch(rect)
        
        # 显示mask提示（如果有）
        if 'mask_input' in prompts:
            # 在标题中标注使用了mask prompt
            name += "_with_mask_prompt"
        
        ax.set_title(name)
        ax.axis('off')
        
        # 添加图例
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc='upper right')
        
        # 保存
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{name}.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"[SAM2] Visualization saved to: {output_path}")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """返回工具能力描述"""
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "model": self.model_id,
            "device": self.device,
            "prompt_types": {
                "point_prompts": {
                    "type": "array",
                    "description": "点坐标列表 [[x1,y1], [x2,y2], ...]",
                    "example": [[100, 200], [300, 400]]
                },
                "point_labels": {
                    "type": "array",
                    "description": "点标签列表 [1,0,...] (1=前景/正样本, 0=背景/负样本)",
                    "example": [1, 0]
                },
                "box_prompt": {
                    "type": "array",
                    "description": "边界框坐标 [x1, y1, x2, y2]",
                    "example": [50, 50, 200, 200]
                },
                "mask_prompt": {
                    "type": "array/numpy.ndarray",
                    "description": "初始分割掩码（二值数组）",
                    "note": "用于refinement或作为初始提示"
                }
            },
            "tasks": {
                "standard": {
                    "description": "使用任意组合的prompts进行分割",
                    "parameters": ["point_prompts", "point_labels", "box_prompt", "mask_prompt", "multimask_output"]
                },
                "refinement": {
                    "description": "基于之前的结果进行精修",
                    "parameters": ["mask_idx", "point_prompts", "point_labels", "box_prompt"],
                    "task": "refinement"
                },
                "smart_medical_segment": {
                    "description": "智能医学图像分割（高级功能）",
                    "parameters": ["question"],
                    "task": "smart_medical_segment"
                }
            },
            "output": {
                "success": {
                    "type": "boolean",
                    "description": "是否成功执行"
                },
                "task": {
                    "type": "string",
                    "description": "执行的任务类型"
                },
                "prompt_types": {
                    "type": "array",
                    "description": "使用的prompt类型列表"
                },
                "results": {
                    "type": "array",
                    "description": "分割结果列表，包含掩码信息、分数、边界框等"
                },
                "best_mask_idx": {
                    "type": "integer",
                    "description": "最佳掩码的索引"
                }
            }
        }