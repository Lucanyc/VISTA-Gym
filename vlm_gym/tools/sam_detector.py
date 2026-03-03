
# vlm_gym/tools/sam_detector.py
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple
from .base import BaseVisionTool

class SAMDetector(BaseVisionTool):
    """使用SAM进行目标检测和分割"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = config.get("model_type", "vit_h")  # vit_h, vit_l, vit_b
        self.checkpoint_path = config.get("checkpoint_path", None)
        
        # 延迟加载模型
        self.predictor = None
        self.grounding_dino = None  # 用于文本引导的检测
        
    def _load_models(self):
        """延迟加载模型"""
        if self.predictor is None:
            try:
                from segment_anything import sam_model_registry, SamPredictor
                
                # 加载SAM模型
                sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
                sam.to(device=self.device)
                self.predictor = SamPredictor(sam)
                
                # 可选：加载GroundingDINO用于文本引导
                try:
                    from groundingdino.util.inference import load_model, predict
                    self.grounding_dino = load_model(
                        "groundingdino/config/GroundingDINO_SwinT_OGC.py",
                        "weights/groundingdino_swint_ogc.pth"
                    )
                except:
                    print("GroundingDINO not available, using box prompt only")
                    
            except ImportError:
                raise ImportError("Please install segment-anything: pip install segment-anything")
    
    def process(self, image: Image.Image, 
                target_text: Optional[str] = None,
                box_prompt: Optional[List[int]] = None,
                point_prompt: Optional[List[Tuple[int, int]]] = None) -> Dict[str, Any]:
        """
        使用SAM检测和分割目标
        
        Args:
            image: PIL图像
            target_text: 文本描述（需要GroundingDINO）
            box_prompt: 边界框提示 [x1, y1, x2, y2]
            point_prompt: 点提示 [(x, y), ...]
        
        Returns:
            包含检测结果的字典
        """
        self._load_models()
        
        # 转换图像
        image_np = np.array(image)
        self.predictor.set_image(image_np)
        
        results = {
            "detections": [],
            "masks": [],
            "status": "success"
        }
        
        # 如果有文本描述，先用GroundingDINO检测
        if target_text and self.grounding_dino:
            boxes, logits, phrases = self._detect_with_text(image_np, target_text)
            
            for i, (box, score, phrase) in enumerate(zip(boxes, logits, phrases)):
                # 使用SAM细化分割
                masks, scores, _ = self.predictor.predict(
                    box=box.numpy(),
                    multimask_output=False
                )
                
                results["detections"].append({
                    "bbox": box.tolist(),
                    "confidence": float(score),
                    "label": phrase,
                    "mask": masks[0]
                })
                results["masks"].append(masks[0])
        
        # 如果有框提示
        elif box_prompt:
            box_np = np.array(box_prompt)
            masks, scores, _ = self.predictor.predict(
                box=box_np,
                multimask_output=False
            )
            
            results["detections"].append({
                "bbox": box_prompt,
                "confidence": float(scores[0]),
                "mask": masks[0]
            })
            results["masks"].append(masks[0])
        
        # 如果有点提示
        elif point_prompt:
            point_coords = np.array(point_prompt)
            point_labels = np.ones(len(point_prompt))  # 1表示前景点
            
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True
            )
            
            # 选择最佳mask
            best_idx = np.argmax(scores)
            results["detections"].append({
                "points": point_prompt,
                "confidence": float(scores[best_idx]),
                "mask": masks[best_idx]
            })
            results["masks"].append(masks[best_idx])
        
        return results
    
    def _detect_with_text(self, image: np.ndarray, text: str) -> Tuple:
        """使用GroundingDINO进行文本引导的检测"""
        # 这里是简化版本，实际需要完整实现
        # 返回边界框、置信度和短语
        return [], [], []