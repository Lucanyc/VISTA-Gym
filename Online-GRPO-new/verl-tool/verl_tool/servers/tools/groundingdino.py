"""
GroundingDINO Tool for VERL Framework - Local Model Version
使用本地下载的 Hugging Face 模型的零样本目标检测工具
Modified to support verl-tool extra_info format
"""
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
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from .base import BaseTool, register_tool

# 设置日志
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. 定义 Ray Actor - 模型的唯一持有者（保持不变）
# ==============================================================================
@ray.remote(num_gpus=1 if torch.cuda.is_available() else 0)
class GroundingDINOActor:
    """Ray Actor that holds the GroundingDINO model"""
    def __init__(self, model_path: str = "/home/meng/model/GroundingDINO"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Initializing GroundingDINOActor on PID: {os.getpid()}, Device: {self.device}")
        
        # 验证模型路径
        if not os.path.exists(model_path):
            self.logger.error(f"Model path does not exist: {model_path}")
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # 验证必要的模型文件
        required_files = ['model.safetensors', 'config.json', 'preprocessor_config.json']
        missing_files = []
        for file in required_files:
            file_path = os.path.join(model_path, file)
            if not os.path.exists(file_path):
                missing_files.append(file)
            elif file == 'model.safetensors':
                # 检查模型文件大小
                file_size = os.path.getsize(file_path)
                if file_size < 1000000:  # 小于1MB可能有问题
                    self.logger.warning(f"Model file seems too small: {file_size} bytes")
        
        if missing_files:
            self.logger.error(f"Missing required model files: {missing_files}")
            raise FileNotFoundError(f"Missing files in {model_path}: {missing_files}")
        
        # 加载本地模型和处理器
        try:
            self.model_path = model_path
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                local_files_only=True  # 确保只使用本地文件
            )
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_path,
                local_files_only=True  # 确保只使用本地文件
            )
            self.model.to(self.device)
            self.model.eval()
            self.logger.info(f"✓ GroundingDINO model loaded successfully from local path: {model_path}")
            
            # 打印模型信息
            model_size = sum(p.numel() for p in self.model.parameters()) / 1e6
            self.logger.info(f"Model size: {model_size:.2f}M parameters")
            
        except Exception as e:
            self.logger.error(f"Failed to load GroundingDINO model from {model_path}: {e}")
            self.processor = None
            self.model = None
    
    def detect(self, image_array: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行目标检测"""
        if not self.model or not self.processor:
            return {"error": "Model not loaded properly"}
        
        try:
            # 参数提取
            query = params.get('query', '')
            box_threshold = params.get('box_threshold', 0.3)
            text_threshold = params.get('text_threshold', 0.25)
            
            # 确保查询格式正确（小写 + 句号结尾）
            if query and not query.endswith('.'):
                query = query + '.'
            query = query.lower()
            
            # 转换图像
            image = Image.fromarray(image_array.astype('uint8'), 'RGB')
            
            # 处理输入
            inputs = self.processor(images=image, text=query, return_tensors="pt").to(self.device)
            
            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 后处理
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]]  # (height, width)
            )[0]  # 只有一张图片，取第一个结果
            
            # 格式化结果
            detections = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                # box 格式: [x1, y1, x2, y2]
                detections.append({
                    "label": label,
                    "confidence": float(score),
                    "bbox": [float(x) for x in box.tolist()]
                })
            
            return {
                "detections": detections,
                "query": query,
                "num_detections": len(detections),
                "model_path": self.model_path,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold
            }
            
        except Exception as e:
            self.logger.error(f"Detection failed: {e}", exc_info=True)
            return {"error": f"Detection failed: {str(e)}"}
    
    def health_check(self) -> str:
        """健康检查"""
        if self.model is not None and self.processor is not None:
            return f"healthy: model loaded from {self.model_path}"
        else:
            return "unhealthy: model not loaded"

# ==============================================================================
# 2. GroundingDINOTool - 修改以支持verl-tool格式
# ==============================================================================
@register_tool
class GroundingDINOTool(BaseTool):
    tool_type = "groundingdino"
    
    def __init__(self, num_workers=1, model_path=None, **kwargs):
        super().__init__(num_workers)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 本地模型路径
        self.model_path = model_path or os.environ.get(
            'GROUNDINGDINO_MODEL_PATH',
            "/home/meng/model/GroundingDINO"
        )
        
        # 验证模型路径存在
        if not os.path.exists(self.model_path):
            self.logger.warning(f"Model path does not exist: {self.model_path}")
        
        self.logger.info(f"Connecting to GroundingDINO Actor with model path: {self.model_path}")
        try:
            # 创建或连接到 Ray Actor
            num_gpus = 1 if torch.cuda.is_available() else 0
            self.actor = GroundingDINOActor.options(
                name="GroundingDINOActor",
                get_if_exists=True,
                num_gpus=num_gpus,
                max_concurrency=10
            ).remote(model_path=self.model_path)
            
            # 健康检查
            self.logger.info("Performing health check...")
            health_status = ray.get(self.actor.health_check.remote(), timeout=60)
            
            if "healthy" in health_status:
                self.logger.info(f"✓ GroundingDINO Actor is {health_status}")
            else:
                raise RuntimeError(f"GroundingDINO Actor health check failed: {health_status}")
                
        except Exception as e:
            self.logger.error(f"Failed to create or connect to GroundingDINO Actor: {e}")
            self.actor = None
    
    def get_usage_inst(self):
        """返回工具使用说明"""
        return 'GroundingDINO: Object detection tool. Use <tool_call>{"tool": "groundingdino", "task": "detect", "query": "object description"}</tool_call>'
    
    def parse_action(self, action: str) -> Tuple[Dict[str, Any], bool]:
        """解析工具调用参数"""
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.findall(tool_call_pattern, action, re.DOTALL)
        
        if not matches:
            return {}, False
        
        try:
            params = json.loads(matches[0].strip())
            
            # 验证是否是针对本工具的调用
            if params.get('tool') != 'groundingdino':
                return {}, False
            
            # 提取参数
            parsed_params = {
                'task': params.get('task', 'detect'),
                'query': params.get('query', ''),
                'box_threshold': float(params.get('box_threshold', 0.3)),
                'text_threshold': float(params.get('text_threshold', 0.25)),
                'raw_params': params
            }
            
            # 验证必需参数
            if not parsed_params['query']:
                self.logger.warning("Missing required parameter: query")
                return {}, False
            
            return parsed_params, True
            
        except Exception as e:
            self.logger.error(f"Error parsing action: {e}")
            return {}, False
    
    def conduct_action(self, trajectory_id: str, action: str, extra_field: Dict[str, Any]) -> Tuple[str, bool, bool]:
        """执行检测操作"""
        parsed_params, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)
        
        if not is_valid:
            self.update_env(trajectory_id, env, action, False, extra_field, "")
            self.save_env(trajectory_id, env)
            return "", False, False
        
        try:
            # 调试日志：查看收到的extra_field
            self.logger.debug(f"[GroundingDINO] Received extra_field keys: {list(extra_field.keys()) if extra_field else 'None'}")
            if extra_field and 'qid' in extra_field:
                self.logger.debug(f"[GroundingDINO] Processing task_id: {extra_field.get('qid')}")
            
            # 获取图像
            image = self._get_image(trajectory_id, extra_field, env)
            if image is None:
                observation = "Error: No image provided for detection"
                self.logger.error(f"[GroundingDINO] {observation}")
                self.logger.debug(f"[GroundingDINO] extra_field content: {extra_field}")
            elif self.actor is None:
                observation = "Error: GroundingDINO Actor is not available"
            else:
                self.logger.debug(f"[GroundingDINO] Successfully loaded image, detecting: {parsed_params['query']}")
                # 转换为 numpy 数组
                image_array = np.array(image.convert('RGB'))
                
                # 调用 Actor 进行检测
                result_ref = self.actor.detect.remote(image_array, parsed_params)
                try:
                    result = ray.get(result_ref, timeout=200)
                    
                    if "error" in result:
                        observation = f"Error: {result['error']}"
                    else:
                        observation = self._format_detections(result)
                        self.logger.debug(f"[GroundingDINO] Detection completed, found {result.get('num_detections', 0)} objects")
                        
                except ray.exceptions.GetTimeoutError:
                    observation = "Error: GroundingDINO detection timed out (30s)"
                    self.logger.error(observation)
            
            self.update_env(trajectory_id, env, parsed_params, is_valid, extra_field, observation)
            self.save_env(trajectory_id, env)
            return f"\n```output\n{observation}\n```\n", False, True
            
        except Exception as e:
            self.logger.error(f"GroundingDINO tool conduct_action failed: {e}", exc_info=True)
            observation = f"Error: {str(e)}"
            return f"\n```output\n{observation}\n```\n", False, True
    
    def _get_image(self, trajectory_id: str, extra_field: Dict[str, Any], env: Dict[str, Any]) -> Optional[Image.Image]:
        """从extra_field或环境中获取图像 - 修改以支持verl-tool格式"""
        
        # 优先处理verl-tool格式的extra_info（包含images列表）
        if extra_field and 'images' in extra_field:
            images_list = extra_field['images']
            self.logger.debug(f"[GroundingDINO] Found 'images' in extra_field: {images_list}")
            
            if images_list:
                # 获取第一个图片路径
                img_path = images_list[0] if isinstance(images_list, list) else images_list
                
                if img_path and isinstance(img_path, str):
                    if os.path.exists(img_path):
                        self.logger.debug(f"[GroundingDINO] Loading image from path: {img_path}")
                        return Image.open(img_path).convert('RGB')
                    else:
                        self.logger.warning(f"[GroundingDINO] Image path does not exist: {img_path}")
                else:
                    self.logger.warning(f"[GroundingDINO] Invalid image path format: {img_path}")
        
        # 向后兼容：原有的image字段处理
        if extra_field and 'image' in extra_field and extra_field['image']:
            img_data = extra_field['image']
            if isinstance(img_data, Image.Image):
                self.logger.debug("[GroundingDINO] Using PIL Image from extra_field['image']")
                return img_data
            if isinstance(img_data, str) and os.path.exists(img_data):
                self.logger.debug(f"[GroundingDINO] Loading image from extra_field['image']: {img_data}")
                return Image.open(img_data).convert('RGB')
        
        # 向后兼容：原有的image_path字段处理
        if extra_field and 'image_path' in extra_field and extra_field['image_path']:
            path_info = extra_field['image_path']
            if isinstance(path_info, list) and path_info:
                img_path = path_info[0].get("image_url") if isinstance(path_info[0], dict) else path_info[0]
                if img_path and os.path.exists(img_path):
                    self.logger.debug(f"[GroundingDINO] Loading image from extra_field['image_path']: {img_path}")
                    return Image.open(img_path).convert('RGB')
        
        # 从环境缓存中获取
        if 'current_image' in env:
            self.logger.debug("[GroundingDINO] Using cached image from environment")
            return env['current_image']
        
        self.logger.warning("[GroundingDINO] No image found in any expected location")
        return None
    
    def _format_detections(self, result: Dict[str, Any]) -> str:
        """格式化检测结果"""
        detections = result.get('detections', [])
        query = result.get('query', '')
        num_detections = result.get('num_detections', 0)
        
        if not detections:
            return f"No objects matching '{query}' were detected in the image."
        
        # 开始格式化输出
        formatted_lines = [f"Detected {num_detections} object(s) matching '{query}':"]
        
        for i, detection in enumerate(detections):
            label = detection.get('label', 'Unknown')
            confidence = detection.get('confidence', 0.0)
            bbox = detection.get('bbox', [])
            
            # bbox 格式: [x1, y1, x2, y2]
            if len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
                width = x2 - x1
                height = y2 - y1
                bbox_str = f"[x1: {x1:.1f}, y1: {y1:.1f}, x2: {x2:.1f}, y2: {y2:.1f}] (w: {width:.1f}, h: {height:.1f})"
            else:
                bbox_str = str(bbox)
            
            formatted_lines.append(
                f"\nObject {i+1}:\n"
                f"  - Label: {label}\n"
                f"  - Confidence: {confidence:.3f}\n"
                f"  - Bounding Box: {bbox_str}"
            )
        
        # 添加统计信息
        if len(detections) > 1:
            avg_confidence = sum(d.get('confidence', 0) for d in detections) / len(detections)
            formatted_lines.append(f"\nAverage confidence: {avg_confidence:.3f}")
            
            # 统计每个标签的数量
            label_counts = {}
            for d in detections:
                label = d.get('label', 'Unknown')
                label_counts[label] = label_counts.get(label, 0) + 1
            
            if len(label_counts) > 1:
                formatted_lines.append("\nLabel distribution:")
                for label, count in sorted(label_counts.items()):
                    formatted_lines.append(f"  - {label}: {count}")
        
        # 添加使用的阈值信息
        if 'box_threshold' in result:
            formatted_lines.append(f"\nDetection thresholds: box={result['box_threshold']}, text={result.get('text_threshold', 0.25)}")
        
        return "\n".join(formatted_lines)
    
    def update_env(self, trajectory_id: str, env: Dict[str, Any], action: Any,
                   is_valid: bool, extra_field: Dict[str, Any], observation: str, **kwargs):
        """更新环境状态"""
        super().update_env(trajectory_id, env, action, is_valid, extra_field, observation, **kwargs)
        # 缓存当前图像
        image_to_cache = self._get_image(trajectory_id, extra_field, env)
        if image_to_cache:
            env['current_image'] = image_to_cache