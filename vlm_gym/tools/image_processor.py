# vlm_gym/tools/image_processor.py
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from .base import BaseVisionTool

class ImageProcessor(BaseVisionTool):
    """图像处理工具：裁剪、缩放、标注等"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
    def process(self, image: Image.Image, **kwargs) -> Dict[str, Any]:
        """处理图像"""
        action = kwargs.get("action", "crop")
        
        if action == "crop":
            return self.crop_image(image, kwargs.get("bbox"))
        elif action == "zoom":
            return self.zoom_region(image, kwargs.get("bbox"), kwargs.get("zoom_factor", 2.0))
        elif action == "annotate":
            return self.annotate_image(image, kwargs.get("annotations", []))
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def crop_image(self, image: Image.Image, bbox: List[int]) -> Dict[str, Any]:
        """裁剪图像到指定区域"""
        if not bbox or len(bbox) != 4:
            return {"status": "error", "message": "Invalid bbox"}
        
        x1, y1, x2, y2 = bbox
        
        # 确保坐标在图像范围内
        width, height = image.size
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))
        
        # 裁剪
        cropped = image.crop((x1, y1, x2, y2))
        
        return {
            "status": "success",
            "image": cropped,
            "original_size": image.size,
            "crop_size": cropped.size,
            "bbox": [x1, y1, x2, y2]
        }
    
    def zoom_region(self, image: Image.Image, bbox: List[int], zoom_factor: float = 2.0) -> Dict[str, Any]:
        """放大指定区域"""
        if not bbox or len(bbox) != 4:
            return {"status": "error", "message": "Invalid bbox"}
        
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # 计算新的边界框（缩小以放大内容）
        new_width = (x2 - x1) / zoom_factor
        new_height = (y2 - y1) / zoom_factor
        
        new_x1 = int(cx - new_width / 2)
        new_y1 = int(cy - new_height / 2)
        new_x2 = int(cx + new_width / 2)
        new_y2 = int(cy + new_height / 2)
        
        # 裁剪并调整大小
        cropped = image.crop((new_x1, new_y1, new_x2, new_y2))
        zoomed = cropped.resize((x2 - x1, y2 - y1), Image.Resampling.LANCZOS)
        
        return {
            "status": "success",
            "image": zoomed,
            "zoom_factor": zoom_factor,
            "original_bbox": bbox,
            "zoom_bbox": [new_x1, new_y1, new_x2, new_y2]
        }
    
    def annotate_image(self, image: Image.Image, annotations: List[Dict]) -> Dict[str, Any]:
        """在图像上添加标注"""
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        
        for ann in annotations:
            if ann["type"] == "box":
                bbox = ann["bbox"]
                color = ann.get("color", "red")
                width = ann.get("width", 2)
                draw.rectangle(bbox, outline=color, width=width)
                
                # 添加标签
                if "label" in ann:
                    draw.text((bbox[0], bbox[1] - 20), ann["label"], fill=color)
            
            elif ann["type"] == "point":
                x, y = ann["coords"]
                radius = ann.get("radius", 5)
                color = ann.get("color", "red")
                draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
        
        return {
            "status": "success",
            "image": annotated,
            "num_annotations": len(annotations)
        }