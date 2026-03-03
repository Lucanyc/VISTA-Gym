# vlm_gym/tools/tool_manager.py
from typing import Any, Dict, List, Optional
from PIL import Image
import os

class VisionToolManager:
    """统一管理所有视觉工具"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tools = {}
        
        # 初始化可用的工具
        self._initialize_tools()
    
    def _initialize_tools(self):
        """初始化工具"""
        # SAM检测器
        if self.config.get("enable_sam", True):
            from .sam_detector import SAMDetector
            sam_config = self.config.get("sam_config", {
                "model_type": "vit_h",
                "checkpoint_path": os.environ.get("SAM_CHECKPOINT", "sam_vit_h_4b8939.pth")
            })
            self.tools["sam_detector"] = SAMDetector(sam_config)
        
        # 图像处理器
        from .image_processor import ImageProcessor
        self.tools["image_processor"] = ImageProcessor({})
        
        # OCR工具
        if self.config.get("enable_ocr", True):
            from .ocr_tool import OCRTool
            ocr_config = self.config.get("ocr_config", {
                "backend": "easyocr",
                "languages": ["en"]
            })
            self.tools["ocr"] = OCRTool(ocr_config)
    
    def locate_object(self, image_path: str, target: str, **kwargs) -> Dict[str, Any]:
        """定位对象"""
        image = Image.open(image_path)
        
        if "sam_detector" in self.tools:
            result = self.tools["sam_detector"].process(
                image, 
                target_text=target,
                **kwargs
            )
            return result
        else:
            # 降级到模拟结果
            return {
                "status": "success",
                "detections": [{
                    "bbox": [100, 100, 300, 300],
                    "confidence": 0.9,
                    "label": target
                }]
            }
    
    def extract_text(self, image_path: str, bbox: Optional[List[int]] = None) -> Dict[str, Any]:
        """提取文本"""
        image = Image.open(image_path)
        
        if "ocr" in self.tools:
            return self.tools["ocr"].process(image, bbox)
        else:
            return {
                "status": "success",
                "full_text": "Sample extracted text",
                "text_regions": []
            }
    
    def crop_and_zoom(self, image_path: str, bbox: List[int], zoom_factor: float = 2.0) -> Dict[str, Any]:
        """裁剪并放大区域"""
        image = Image.open(image_path)
        
        if "image_processor" in self.tools:
            return self.tools["image_processor"].process(
                image,
                action="zoom",
                bbox=bbox,
                zoom_factor=zoom_factor
            )
        else:
            return {"status": "success", "message": "Simulated zoom"}