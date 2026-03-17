import json
import re
import os
import sys
import logging
from typing import Tuple, Dict, Any, Optional, List
import numpy as np

import ray
import torch
from pathlib import Path
from PIL import Image

from .base import BaseTool, register_tool

# 设置日志
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. 定义 Ray Actor - 模型的唯一持有者（保持不变）
# ==============================================================================
@ray.remote(num_gpus=1 if torch.cuda.is_available() else 0)
class EasyOCRActor:
    """Ray Actor that holds the EasyOCR model"""
    def __init__(self, model_path: str = "/data/models/EasyOCR"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Initializing EasyOCRActor on PID: {os.getpid()}, Device: {self.device}")
        
        # 加载模型
        try:
            # 将 EasyOCR 路径添加到 Python 路径
            sys.path.insert(0, model_path)
            import easyocr
            
            # 初始化 reader，默认支持英文和简体中文
            self.readers = {}
            self.model_path = model_path
            
            # 预加载常用语言组合
            common_langs = [['en'], ['ch_sim'], ['en', 'ch_sim']]
            for langs in common_langs:
                lang_key = ','.join(sorted(langs))
                self.readers[lang_key] = easyocr.Reader(
                    langs, 
                    gpu=torch.cuda.is_available(),
                    model_storage_directory=os.path.join(model_path, '.EasyOCR'),
                    download_enabled=True  # 使用本地模型
                )
                self.logger.info(f"✓ EasyOCR reader loaded for languages: {langs}")
                
        except Exception as e:
            self.logger.error(f"Failed to load EasyOCR model: {e}")
            self.readers = {}
    
    def _get_reader(self, languages: List[str]):
        """获取或创建指定语言的 reader"""
        lang_key = ','.join(sorted(languages))
        
        if lang_key not in self.readers:
            self.logger.info(f"Creating new reader for languages: {languages}")
            try:
                sys.path.insert(0, self.model_path)
                import easyocr
                
                self.readers[lang_key] = easyocr.Reader(
                    languages,
                    gpu=torch.cuda.is_available(),
                    model_storage_directory=os.path.join(self.model_path, '.EasyOCR'),
                    download_enabled=False
                )
            except Exception as e:
                self.logger.error(f"Failed to create reader for {languages}: {e}")
                return None
                
        return self.readers[lang_key]
    
    def ocr(self, image_array: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行 OCR 识别"""
        if not self.readers:
            return {"error": "No EasyOCR readers loaded"}
        
        try:
            # 参数提取
            languages = params.get('languages', ['en'])
            detail = params.get('detail', 1)  # 0=simple, 1=detailed
            paragraph = params.get('paragraph', False)
            
            # 获取对应语言的 reader
            reader = self._get_reader(languages)
            if reader is None:
                return {"error": f"Failed to load reader for languages: {languages}"}
            
            # 执行 OCR
            results = reader.readtext(
                image_array,
                detail=detail,
                paragraph=paragraph
            )
            
            # 格式化结果
            if detail == 0:
                # Simple format: just text list
                text_list = results
                return {
                    "texts": text_list,
                    "num_texts": len(text_list),
                    "languages": languages
                }
            else:
                # Detailed format: list of (bbox, text, confidence)
                detections = []
                for result in results:
                    if len(result) >= 3:
                        bbox, text, confidence = result[:3]
                        detections.append({
                            "text": text,
                            "confidence": float(confidence),
                            "bbox": [[float(x), float(y)] for x, y in bbox]
                        })
                
                return {
                    "detections": detections,
                    "num_detections": len(detections),
                    "languages": languages,
                    "paragraph": paragraph
                }
                
        except Exception as e:
            self.logger.error(f"OCR failed: {e}", exc_info=True)
            return {"error": f"OCR failed: {str(e)}"}
    
    def health_check(self) -> str:
        """健康检查"""
        if self.readers:
            num_readers = len(self.readers)
            return f"healthy: {num_readers} reader(s) loaded"
        else:
            return "unhealthy: no readers loaded"

# ==============================================================================
# 2. EasyOCRTool - 修改以支持verl-tool格式
# ==============================================================================
@register_tool
class EasyOCRTool(BaseTool):
    tool_type = "easyocr"
    
    def __init__(self, num_workers=1, model_path=None, **kwargs):
        super().__init__(num_workers)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 模型路径
        self.model_path = model_path or "/data/models/EasyOCR"
        
        self.logger.info("Connecting to EasyOCR Actor...")
        try:
            # 创建或连接到 Ray Actor
            num_gpus = 1 if torch.cuda.is_available() else 0
            self.actor = EasyOCRActor.options(
                name="EasyOCRActor",
                get_if_exists=True,
                num_gpus=num_gpus,
                max_concurrency=10
            ).remote(model_path=self.model_path)
            
            # 健康检查
            self.logger.info("Performing health check...")
            health_status = ray.get(self.actor.health_check.remote(), timeout=60)
            
            if "healthy" in health_status:
                self.logger.info(f"✓ EasyOCR Actor is {health_status}")
            else:
                raise RuntimeError(f"EasyOCR Actor health check failed: {health_status}")
                
        except Exception as e:
            self.logger.error(f"Failed to create or connect to EasyOCR Actor: {e}")
            self.actor = None
    
    def get_usage_inst(self):
        """返回工具使用说明"""
        return 'EasyOCR: Text extraction tool. Use <tool_call>{"tool": "easyocr", "task": "ocr", "languages": ["en"]}</tool_call>'
    
    def parse_action(self, action: str) -> Tuple[Dict[str, Any], bool]:
        """解析工具调用参数"""
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.findall(tool_call_pattern, action, re.DOTALL)
        
        if not matches:
            return {}, False
        
        try:
            params = json.loads(matches[0].strip())
            
            # 验证是否是针对本工具的调用
            if params.get('tool') != 'easyocr':
                return {}, False
            
            # 提取参数
            parsed_params = {
                'task': params.get('task', 'ocr'),
                'languages': params.get('languages', ['en']),
                'detail': int(params.get('detail', 1)),
                'paragraph': bool(params.get('paragraph', False)),
                'raw_params': params
            }
            
            # 验证语言参数
            if not isinstance(parsed_params['languages'], list):
                parsed_params['languages'] = [parsed_params['languages']]
            
            return parsed_params, True
            
        except Exception as e:
            logger.error(f"Error parsing action: {e}")
            return {}, False
    
    def conduct_action(self, trajectory_id: str, action: str, extra_field: Dict[str, Any]) -> Tuple[str, bool, bool]:
        """执行 OCR 操作"""
        # ===== DEBUG INFO =====
        print(f"\n{'='*70}")
        print(f"[EasyOCR conduct_action] Called")
        print(f"  trajectory_id: {trajectory_id}")
        print(f"  action: {action[:100]}...")
        print(f"  extra_field type: {type(extra_field)}")
        print(f"  extra_field is None: {extra_field is None}")
        if extra_field:
            print(f"  extra_field keys: {list(extra_field.keys())}")
            if 'images' in extra_field:
                print(f"  extra_field['images']: {extra_field['images']}")
        else:
            print(f"  WARNING: extra_field is None or empty!")
        print(f"{'='*70}\n")
        
        parsed_params, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)
        
        if not is_valid:
            self.update_env(trajectory_id, env, action, False, extra_field, "")
            self.save_env(trajectory_id, env)
            return "", False, False
        
        try:
            # 调试日志：查看收到的extra_field
            self.logger.debug(f"[EasyOCR] Received extra_field keys: {list(extra_field.keys()) if extra_field else 'None'}")
            if extra_field and 'qid' in extra_field:
                self.logger.debug(f"[EasyOCR] Processing task_id: {extra_field.get('qid')}")
            
            # 获取图像
            image = self._get_image(trajectory_id, extra_field, env)
            if image is None:
                observation = f"Error: No image provided. extra_field keys: {list(extra_field.keys()) if extra_field else 'None'}, type: {type(extra_field)}"
                self.logger.error(f"[EasyOCR] {observation}")
                self.logger.debug(f"[EasyOCR] extra_field content: {extra_field}")
            elif self.actor is None:
                observation = "Error: EasyOCR Actor is not available"
            else:
                self.logger.debug(f"[EasyOCR] Successfully loaded image, performing OCR with languages: {parsed_params['languages']}")
                # 转换为 numpy 数组
                image_array = np.array(image.convert('RGB'))
                
                # 调用 Actor 进行 OCR
                result_ref = self.actor.ocr.remote(image_array, parsed_params)
                try:
                    result = ray.get(result_ref, timeout=200)
                    
                    if "error" in result:
                        observation = f"Error: {result['error']}"
                    else:
                        observation = self._format_ocr_results(result, parsed_params.get('detail', 1))
                        num_detected = result.get('num_detections', 0) if parsed_params.get('detail', 1) else result.get('num_texts', 0)
                        self.logger.debug(f"[EasyOCR] OCR completed, detected {num_detected} text region(s)")
                        
                except ray.exceptions.GetTimeoutError:
                    observation = "Error: EasyOCR processing timed out (30s)"
                    self.logger.error(observation)
            
            self.update_env(trajectory_id, env, parsed_params, is_valid, extra_field, observation)
            self.save_env(trajectory_id, env)
            return f"\n```output\n{observation}\n```\n", False, True
            
        except Exception as e:
            self.logger.error(f"EasyOCR tool conduct_action failed: {e}", exc_info=True)
            observation = f"Error: {str(e)}"
            return f"\n```output\n{observation}\n```\n", False, True
    
    def _get_image(self, trajectory_id: str, extra_field: Dict[str, Any], env: Dict[str, Any]) -> Optional[Image.Image]:
        """从extra_field或环境中获取图像 - 修改以支持verl-tool格式"""
        
        # ===== DEBUG INFO =====
        print(f"\n[_get_image] Called")
        print(f"  trajectory_id: {trajectory_id}")
        print(f"  extra_field: {extra_field}")
        if extra_field and 'images' in extra_field:
            print(f"  Found 'images' field: {extra_field['images']}")
        
        # 优先处理verl-tool格式的extra_info（包含images列表）
        if extra_field and 'images' in extra_field:
            images_list = extra_field['images']
            self.logger.debug(f"[EasyOCR] Found 'images' in extra_field: {images_list}")
            print(f"  [_get_image] Trying to load from 'images' field...")
            
            if images_list:
                # 获取第一个图片路径
                img_path = images_list[0] if isinstance(images_list, list) else images_list
                
                if img_path and isinstance(img_path, str):
                    print(f"  [_get_image] Checking path: {img_path}")
                    if os.path.exists(img_path):
                        print(f"  [_get_image] ✅ Path exists, loading image...")
                        self.logger.debug(f"[EasyOCR] Loading image from path: {img_path}")
                        try:
                            img = Image.open(img_path).convert('RGB')
                            print(f"  [_get_image] ✅ Successfully loaded image: {img.size}")
                            return img
                        except Exception as e:
                            print(f"  [_get_image] ❌ Failed to load image: {e}")
                    else:
                        print(f"  [_get_image] ❌ Path does not exist: {img_path}")
                        self.logger.warning(f"[EasyOCR] Image path does not exist: {img_path}")
                else:
                    print(f"  [_get_image] ❌ Invalid image path format: {img_path}")
                    self.logger.warning(f"[EasyOCR] Invalid image path format: {img_path}")
        
        # 向后兼容：原有的image字段处理
        if extra_field and 'image' in extra_field and extra_field['image']:
            print(f"  [_get_image] Trying 'image' field...")
            img_data = extra_field['image']
            if isinstance(img_data, Image.Image):
                self.logger.debug("[EasyOCR] Using PIL Image from extra_field['image']")
                print(f"  [_get_image] ✅ Found PIL Image")
                return img_data
            if isinstance(img_data, str) and os.path.exists(img_data):
                self.logger.debug(f"[EasyOCR] Loading image from extra_field['image']: {img_data}")
                print(f"  [_get_image] ✅ Loading from image path")
                return Image.open(img_data).convert('RGB')
        
        # 向后兼容：原有的image_path字段处理
        if extra_field and 'image_path' in extra_field and extra_field['image_path']:
            print(f"  [_get_image] Trying 'image_path' field...")
            path_info = extra_field['image_path']
            if isinstance(path_info, list) and path_info:
                img_path = path_info[0].get("image_url") if isinstance(path_info[0], dict) else path_info[0]
                if img_path and os.path.exists(img_path):
                    self.logger.debug(f"[EasyOCR] Loading image from extra_field['image_path']: {img_path}")
                    print(f"  [_get_image] ✅ Loading from image_path")
                    return Image.open(img_path).convert('RGB')
        
        # 从环境缓存中获取
        if 'current_image' in env:
            self.logger.debug("[EasyOCR] Using cached image from environment")
            print(f"  [_get_image] Using cached image from environment")
            return env['current_image']
        
        print(f"  [_get_image] ❌ No image found in any location!")
        self.logger.warning("[EasyOCR] No image found in any expected location")
        return None
    
    def _format_ocr_results(self, result: Dict[str, Any], detail: int) -> str:
        """格式化 OCR 结果"""
        languages = result.get('languages', ['unknown'])
        lang_str = ', '.join(languages)
        
        if detail == 0:
            # Simple format
            texts = result.get('texts', [])
            if not texts:
                return f"No text detected in the image (languages: {lang_str})."
            
            formatted_lines = [f"Extracted text (languages: {lang_str}):"]
            for i, text in enumerate(texts):
                formatted_lines.append(f"{i+1}. {text}")
            
            return "\n".join(formatted_lines)
        
        else:
            # Detailed format
            detections = result.get('detections', [])
            if not detections:
                return f"No text detected in the image (languages: {lang_str})."
            
            formatted_lines = [f"Detected {len(detections)} text region(s) (languages: {lang_str}):"]
            
            for i, detection in enumerate(detections):
                text = detection.get('text', '')
                confidence = detection.get('confidence', 0.0)
                bbox = detection.get('bbox', [])
                
                # 格式化边界框
                if bbox and len(bbox) >= 4:
                    bbox_str = f"[{', '.join([f'({x:.1f}, {y:.1f})' for x, y in bbox[:4]])}]"
                else:
                    bbox_str = "N/A"
                
                formatted_lines.append(
                    f"\nText {i+1}:\n"
                    f"  - Content: \"{text}\"\n"
                    f"  - Confidence: {confidence:.3f}\n"
                    f"  - Bounding Box: {bbox_str}"
                )
            
            # 添加统计信息
            if len(detections) > 1:
                avg_confidence = sum(d.get('confidence', 0) for d in detections) / len(detections)
                formatted_lines.append(f"\nAverage confidence: {avg_confidence:.3f}")
                
                # 合并所有文本
                all_text = ' '.join([d.get('text', '') for d in detections])
                formatted_lines.append(f"\nCombined text: \"{all_text}\"")
            
            return "\n".join(formatted_lines)
    
    def update_env(self, trajectory_id: str, env: Dict[str, Any], action: Any,
                   is_valid: bool, extra_field: Dict[str, Any], observation: str, **kwargs):
        """更新环境状态"""
        super().update_env(trajectory_id, env, action, is_valid, extra_field, observation, **kwargs)
        # 缓存当前图像
        image_to_cache = self._get_image(trajectory_id, extra_field, env)
        if image_to_cache:
            env['current_image'] = image_to_cache