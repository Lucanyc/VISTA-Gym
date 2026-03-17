"""
SAM2 Tool for VERL Framework
Image segmentation tool using Segment Anything Model 2 (SAM2)
"""

import sys
import json
import re
import os
import logging
from typing import Tuple, Dict, Any, Optional, List
import numpy as np

import ray
import torch
from pathlib import Path
from PIL import Image

# 设置环境变量
os.environ["HF_TORCH_LOAD_DISABLE_SAFE_CHECK"] = "1"
os.environ["TRUST_REMOTE_CODE"] = "1"

from .base import BaseTool, register_tool

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. 定义 Ray Actor - SAM2模型的持有者
# ==============================================================================
@ray.remote(num_gpus=1 if torch.cuda.is_available() else 0)
class SAM2Actor:
    """Ray Actor for SAM2 model inference"""
    
    def __init__(self, 
                 sam2_dir: str = None,
                 checkpoint_path: str = None, 
                 model_cfg: str = None,
                 model_size: str = "large"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Initializing SAM2Actor on PID: {os.getpid()}, Device: {self.device}")
        
        # SAM2目录路径
        self.sam2_dir = sam2_dir or os.environ.get(
            'SAM2_DIR',
            "/projects/slmreasoning/meng/model/sam2"
        )
        
        # 模型大小配置映射
        self.model_configs = {
            "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
            "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
            "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
            "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
        }
        
        self.model_size = model_size
        self.checkpoint_path = checkpoint_path
        self.model_cfg = model_cfg
        
        # 加载模型
        self.predictor = self._load_model()
        
        if self.predictor:
            self.logger.info("✓ SAM2 Actor initialized successfully.")
        else:
            self.logger.error("❌ SAM2 Actor failed to initialize model.")
    
    def _load_model(self):
        """Load SAM2 model"""
        self.logger.info("Loading SAM2 model...")
        
        try:
            # 添加SAM2到sys.path
            if self.sam2_dir not in sys.path:
                sys.path.insert(0, self.sam2_dir)
            
            # 保存原始工作目录
            original_cwd = os.getcwd()
            
            try:
                # 切换到SAM2目录（Hydra需要）
                os.chdir(self.sam2_dir)
                self.logger.info(f"Changed working directory to: {os.getcwd()}")
                
                # 手动初始化Hydra（类似BiomedParse的方式）
                from hydra import compose
                from hydra.core.global_hydra import GlobalHydra
                from omegaconf import OmegaConf
                
                # 确定配置和checkpoint
                if self.model_size in self.model_configs:
                    config_file, ckpt_name = self.model_configs[self.model_size]
                else:
                    config_file, ckpt_name = self.model_configs["large"]
                    self.logger.warning(f"Unknown model size '{self.model_size}', using 'large'")
                
                # 确定checkpoint路径
                if self.checkpoint_path and os.path.exists(self.checkpoint_path):
                    ckpt_path = self.checkpoint_path
                else:
                    ckpt_path = os.path.join(self.sam2_dir, "checkpoints", ckpt_name)
                
                # 确定配置文件路径
                if self.model_cfg:
                    config_file = self.model_cfg
                
                self.logger.info(f"Using config: {config_file}")
                self.logger.info(f"Using checkpoint: {ckpt_path}")
                
                if not os.path.exists(ckpt_path):
                    self.logger.error(f"Checkpoint not found: {ckpt_path}")
                    return None
                
                # 清除并重新初始化Hydra
                GlobalHydra.instance().clear()
                
                # 使用绝对路径初始化Hydra config
                import hydra
                config_abs_path = os.path.join(self.sam2_dir, "sam2")
                self.logger.info(f"Initializing Hydra with config module path: {config_abs_path}")
                
                hydra.initialize_config_module(config_module="sam2", version_base="1.2")
                
                # 构建模型配置
                cfg = compose(config_name=config_file)
                OmegaConf.resolve(cfg)
                
                # 实例化模型
                model = hydra.utils.instantiate(cfg.model, _recursive_=True)
                
                # 加载checkpoint
                sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
                missing_keys, unexpected_keys = model.load_state_dict(sd)
                if missing_keys:
                    self.logger.warning(f"Missing keys: {missing_keys}")
                if unexpected_keys:
                    self.logger.warning(f"Unexpected keys: {unexpected_keys}")
                
                model = model.to(self.device)
                model.eval()
                
                self.logger.info("✓ SAM2 model loaded successfully")
                
                # 清除Hydra
                GlobalHydra.instance().clear()
                
                # 导入并创建predictor
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                predictor = SAM2ImagePredictor(model)
                
                self.logger.info("✓ SAM2 predictor created successfully")
                return predictor
                
            finally:
                # 恢复原始工作目录
                os.chdir(original_cwd)
                self.logger.info(f"Restored working directory to: {os.getcwd()}")
            
        except Exception as e:
            self.logger.error(f"Failed to load SAM2 model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def inference(self, image_data: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SAM2 inference"""
        image_hash = hash(image_data.tobytes()[:1000])
        self.logger.info(f"[SAM2] Processing image hash: {image_hash}, shape={image_data.shape}, params={params}")
        
        if not self.predictor:
            return {"error": "SAM2 model is not available in the Actor.", "success": False}
        
        try:
            # 获取参数
            task = params.get('task', 'segment')
            point_coords = params.get('point_coords', None)  # [[x, y], ...]
            point_labels = params.get('point_labels', None)  # [1, 0, ...]  1=foreground, 0=background
            box = params.get('box', None)  # [x1, y1, x2, y2]
            boxes = params.get('boxes', None)  # [[x1, y1, x2, y2], ...]
            multimask_output = params.get('multimask_output', True)
            auto_mode = params.get('auto_mode', False)  # 自动分割整个图像
            
            self.logger.info(f"[SAM2] Task: {task}, point_coords: {point_coords}, box: {box}, auto_mode: {auto_mode}")
            
            # 确保图像是RGB格式
            if len(image_data.shape) == 2:
                image_data = np.stack([image_data] * 3, axis=-1)
            elif image_data.shape[-1] == 1:
                image_data = np.concatenate([image_data] * 3, axis=-1)
            
            # 设置图像
            self.predictor.set_image(image_data)
            
            if auto_mode:
                # 自动分割模式：使用AutomaticMaskGenerator
                result = self._auto_segment(image_data)
            else:
                # 交互式分割模式
                # 处理点坐标
                np_point_coords = None
                np_point_labels = None
                if point_coords:
                    np_point_coords = np.array(point_coords, dtype=np.float32)
                    if point_labels:
                        np_point_labels = np.array(point_labels, dtype=np.int32)
                    else:
                        # 默认都是前景点
                        np_point_labels = np.ones(len(point_coords), dtype=np.int32)
                
                # 处理边界框
                np_box = None
                if box:
                    np_box = np.array(box, dtype=np.float32)
                elif boxes:
                    # 如果提供多个box，取第一个
                    np_box = np.array(boxes[0], dtype=np.float32)
                
                # 如果没有任何prompt，使用图像中心点
                if np_point_coords is None and np_box is None:
                    h, w = image_data.shape[:2]
                    np_point_coords = np.array([[w // 2, h // 2]], dtype=np.float32)
                    np_point_labels = np.array([1], dtype=np.int32)
                    self.logger.info(f"[SAM2] No prompt provided, using center point: {np_point_coords}")
                
                # 运行预测
                masks, scores, low_res_masks = self.predictor.predict(
                    point_coords=np_point_coords,
                    point_labels=np_point_labels,
                    box=np_box,
                    multimask_output=multimask_output,
                )
                
                # 构建结果
                result = {
                    "masks_shape": list(masks.shape),
                    "num_masks": masks.shape[0],
                    "scores": scores.tolist(),
                    "best_mask_idx": int(np.argmax(scores)),
                    "best_score": float(np.max(scores)),
                    "task": task,
                    "success": True,
                    "masks": masks.tolist() if masks.size < 1000000 else None,
                    "masks_summary": {
                        "min": float(masks.min()),
                        "max": float(masks.max()),
                        "mean": float(masks.mean()),
                        "has_detections": bool(masks.sum() > 0),
                        "mask_areas": [int(m.sum()) for m in masks],
                    },
                    "input_prompts": {
                        "point_coords": point_coords,
                        "point_labels": point_labels,
                        "box": box,
                    }
                }
            
            # 重置predictor状态
            self.predictor.reset_predictor()
            
            self.logger.debug(f"[SAM2] Segmentation completed, masks shape: {result.get('masks_shape')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"SAM2 inference failed: {e}", exc_info=True)
            return {"error": str(e), "success": False}
    
    def _auto_segment(self, image_data: np.ndarray) -> Dict[str, Any]:
        """自动分割整个图像（使用AutomaticMaskGenerator）"""
        try:
            # 尝试导入AutomaticMaskGenerator
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            
            # 创建自动mask生成器
            mask_generator = SAM2AutomaticMaskGenerator(self.predictor.model)
            
            # 生成masks
            masks_data = mask_generator.generate(image_data)
            
            # 处理结果
            all_masks = []
            all_scores = []
            all_areas = []
            
            for mask_info in masks_data:
                all_masks.append(mask_info['segmentation'])
                all_scores.append(mask_info['predicted_iou'])
                all_areas.append(mask_info['area'])
            
            result = {
                "num_masks": len(masks_data),
                "scores": all_scores,
                "areas": all_areas,
                "task": "auto_segment",
                "success": True,
                "masks_summary": {
                    "total_objects": len(masks_data),
                    "avg_score": float(np.mean(all_scores)) if all_scores else 0,
                    "total_area": sum(all_areas),
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Auto segmentation failed: {e}")
            return {"error": f"Auto segmentation failed: {str(e)}", "success": False}
    
    def health_check(self) -> str:
        """健康检查"""
        if self.predictor is not None:
            return f"healthy: SAM2 model loaded (size={self.model_size})"
        else:
            return "unhealthy: model not loaded"


# ==============================================================================
# 2. SAM2Tool - 工具接口
# ==============================================================================
@register_tool
class SAM2Tool(BaseTool):
    tool_type = "sam2"
    
    def __init__(self, 
                 num_workers=1, 
                 sam2_dir=None,
                 checkpoint_path=None, 
                 model_cfg=None,
                 model_size="large",
                 **kwargs):
        super().__init__(num_workers)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 获取配置
        self.sam2_dir = sam2_dir or os.environ.get(
            'SAM2_DIR',
            "/projects/slmreasoning/meng/model/sam2"
        )
        self.checkpoint_path = checkpoint_path or os.environ.get('SAM2_CHECKPOINT_PATH', None)
        self.model_cfg = model_cfg
        self.model_size = model_size
        
        self.logger.info(f"SAM2 directory: {self.sam2_dir}")
        self.logger.info(f"Model size: {self.model_size}")
        
        # ============================================================
        # 创建Actor
        # ============================================================
        self.logger.info("Connecting to SAM2 Actor...")
        try:
            num_gpus = 1 if torch.cuda.is_available() else 0
            self.actor = SAM2Actor.options(
                name="SAM2Actor",
                get_if_exists=True,
                num_gpus=num_gpus,
                max_concurrency=3
            ).remote(
                sam2_dir=self.sam2_dir,
                checkpoint_path=self.checkpoint_path,
                model_cfg=self.model_cfg,
                model_size=self.model_size
            )
            
            self.logger.info("Performing health check...")
            health_status = ray.get(self.actor.health_check.remote(), timeout=300)
            
            if "healthy" in health_status:
                self.logger.info(f"✓ SAM2 Actor is {health_status}")
            else:
                self.logger.error(f"SAM2 Actor health check failed: {health_status}")
                
        except Exception as e:
            self.logger.error(f"Failed to create or connect to SAM2 Actor: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.actor = None
    
    def get_usage_inst(self):
        """返回工具使用说明"""
        return ('SAM2: Image segmentation tool using Segment Anything Model 2. '
                'Use <tool_call>{"tool": "sam2", "task": "segment", '
                '"point_coords": [[x, y]], "point_labels": [1], '
                '"box": [x1, y1, x2, y2], "multimask_output": true, '
                '"auto_mode": false}</tool_call>. '
                'point_labels: 1=foreground, 0=background. '
                'box format: [x1, y1, x2, y2] in XYXY format.')
    
    def parse_action(self, action: str) -> Tuple[Dict[str, Any], bool]:
        """解析工具调用参数"""
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.findall(tool_call_pattern, action, re.DOTALL)
        
        if not matches:
            return {}, False
        
        try:
            params = json.loads(matches[0].strip())
            
            # 验证工具名
            if params.get('tool') != 'sam2' and params.get('name') != 'sam2':
                return {}, False
            
            # 支持嵌套格式
            if 'parameters' in params and isinstance(params['parameters'], dict):
                inner_params = params['parameters']
            elif 'arguments' in params and isinstance(params['arguments'], dict):
                inner_params = params['arguments']
            else:
                inner_params = params
            
            # 提取所有可能的参数
            parsed = {
                'task': inner_params.get('task', 'segment'),
                'point_coords': inner_params.get('point_coords', None),
                'point_labels': inner_params.get('point_labels', None),
                'box': inner_params.get('box', None),
                'boxes': inner_params.get('boxes', None),
                'multimask_output': inner_params.get('multimask_output', True),
                'auto_mode': inner_params.get('auto_mode', False),
            }
            
            self.logger.debug(f"Parsed parameters: task={parsed['task']}, "
                            f"point_coords={parsed['point_coords']}, "
                            f"box={parsed['box']}")
            
            return parsed, True
            
        except Exception as e:
            self.logger.error(f"Error parsing action: {e}")
            return {}, False
    
    def conduct_action(self, trajectory_id: str, action: str, extra_field: Dict[str, Any]) -> Tuple[str, bool, bool]:
        """执行SAM2工具调用"""
        parsed_params, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)
        
        if not is_valid:
            observation = "Error: Invalid action format. Expected <tool_call>{...}</tool_call>"
            self.update_env(trajectory_id, env, action, False, extra_field, observation)
            self.save_env(trajectory_id, env)
            return f"\n```output\n{observation}\n```\n", False, False
        
        try:
            # 调试日志
            self.logger.info(f"[SAM2] Processing request - trajectory_id: {trajectory_id}")
            if extra_field and 'qid' in extra_field:
                self.logger.info(f"[SAM2] Processing task_id: {extra_field.get('qid')}")
            
            # 获取图像
            image = self._get_image(trajectory_id, extra_field)
            
            if image is None:
                observation = "Error: No image provided for SAM2 analysis"
                self.logger.error(f"[SAM2] {observation}")
                self.logger.error(f"[SAM2] Available extra_field keys: {list(extra_field.keys()) if extra_field else 'None'}")
            elif self.actor is None:
                observation = "Error: SAM2 Actor is not available"
                self.logger.error(observation)
            else:
                self.logger.info(f"[SAM2] Processing with task: {parsed_params['task']}")
                
                image_array = np.array(image.convert('RGB'))
                
                # 传递参数到Actor
                result_ref = self.actor.inference.remote(image_array, parsed_params)
                try:
                    result = ray.get(result_ref, timeout=300)
                    
                    if result.get('success', False):
                        observation = self._format_output(result, parsed_params['task'])
                    else:
                        observation = f"Error: {result.get('error', 'Unknown error')}"
                    
                    self.logger.info(f"[SAM2] Successfully analyzed image for trajectory {trajectory_id}")
                    
                except ray.exceptions.GetTimeoutError:
                    observation = "Error: SAM2 inference timed out (300s)"
                    self.logger.error(observation)
            
            self.update_env(trajectory_id, env, parsed_params, is_valid, extra_field, observation)
            self.save_env(trajectory_id, env)
            return f"\n```output\n{observation}\n```\n", False, True
            
        except Exception as e:
            self.logger.error(f"SAM2 tool conduct_action failed: {e}", exc_info=True)
            observation = f"Error: {str(e)}"
            return f"\n```output\n{observation}\n```\n", False, True
    
    def _get_image(self, trajectory_id: str, extra_field: Dict[str, Any]) -> Optional[Image.Image]:
        """从extra_field获取图像"""
        
        # 优先从extra_field['images']获取
        if extra_field and 'images' in extra_field:
            images_list = extra_field['images']
            self.logger.info(f"[SAM2] Found 'images' in extra_field: {images_list}")
            
            if images_list:
                img_path = images_list[0] if isinstance(images_list, list) else images_list
                
                if img_path and isinstance(img_path, str):
                    if os.path.exists(img_path):
                        self.logger.info(f"[SAM2] ✅ Loading image from: {img_path}")
                        loaded_image = Image.open(img_path).convert('RGB')
                        self.logger.info(f"[SAM2] Image size: {loaded_image.size}")
                        return loaded_image
                    else:
                        self.logger.error(f"[SAM2] ❌ Image path does not exist: {img_path}")
                else:
                    self.logger.error(f"[SAM2] ❌ Invalid image path format: {type(img_path)} - {img_path}")
            else:
                self.logger.error(f"[SAM2] ❌ images list is empty")
        
        # 向后兼容：原有的image字段处理
        if extra_field and 'image' in extra_field and extra_field['image']:
            img_data = extra_field['image']
            if isinstance(img_data, Image.Image):
                self.logger.info("[SAM2] Using PIL Image from extra_field['image']")
                return img_data
            if isinstance(img_data, str) and os.path.exists(img_data):
                self.logger.info(f"[SAM2] Loading image from extra_field['image']: {img_data}")
                return Image.open(img_data).convert('RGB')
        
        self.logger.error(f"[SAM2] ❌ NO IMAGE PROVIDED!")
        self.logger.error(f"[SAM2] Available extra_field keys: {list(extra_field.keys()) if extra_field else 'None'}")
        return None
    
    def _format_output(self, result: Dict[str, Any], task: str) -> str:
        """格式化输出结果"""
        if 'error' in result:
            return f"Error: {result['error']}"
        
        output_lines = []
        
        if task == 'auto_segment' or result.get('task') == 'auto_segment':
            # 自动分割结果
            output_lines.append(f"Auto Segmentation Results:")
            output_lines.append(f"  Total objects detected: {result.get('num_masks', 0)}")
            summary = result.get('masks_summary', {})
            if summary:
                output_lines.append(f"  Average confidence: {summary.get('avg_score', 0):.4f}")
                output_lines.append(f"  Total segmented area: {summary.get('total_area', 0)} pixels")
        else:
            # 交互式分割结果
            output_lines.append(f"Segmentation Results:")
            output_lines.append(f"  Masks shape: {result.get('masks_shape', 'N/A')}")
            output_lines.append(f"  Number of masks: {result.get('num_masks', 0)}")
            output_lines.append(f"  Best mask index: {result.get('best_mask_idx', 0)}")
            output_lines.append(f"  Best score: {result.get('best_score', 0):.4f}")
            
            # 所有mask的分数
            scores = result.get('scores', [])
            if scores:
                output_lines.append(f"  All scores: {[f'{s:.4f}' for s in scores]}")
            
            # 检测摘要
            summary = result.get('masks_summary', {})
            if summary:
                output_lines.append(f"\nMask Statistics:")
                output_lines.append(f"  Has detections: {summary.get('has_detections', False)}")
                mask_areas = summary.get('mask_areas', [])
                if mask_areas:
                    output_lines.append(f"  Mask areas (pixels): {mask_areas}")
            
            # 输入prompts
            input_prompts = result.get('input_prompts', {})
            if input_prompts:
                output_lines.append(f"\nInput prompts used:")
                if input_prompts.get('point_coords'):
                    output_lines.append(f"  Points: {input_prompts['point_coords']}")
                    output_lines.append(f"  Labels: {input_prompts.get('point_labels', [])}")
                if input_prompts.get('box'):
                    output_lines.append(f"  Box: {input_prompts['box']}")
        
        return '\n'.join(output_lines)
    
    def update_env(self, trajectory_id: str, env: Dict[str, Any], action: Any, 
                   is_valid: bool, extra_field: Dict[str, Any], observation: str, **kwargs):
        """更新环境"""
        super().update_env(trajectory_id, env, action, is_valid, extra_field, observation, **kwargs)