
# vlm_gym/tools/ocr_tool.py
from PIL import Image
import numpy as np
from typing import Dict, Any, List, Optional
from .base import BaseVisionTool

class OCRTool(BaseVisionTool):
    """OCR文本提取工具"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.backend = config.get("backend", "easyocr")  # easyocr, paddleocr, tesseract
        self.languages = config.get("languages", ["en", "ch"])
        self.reader = None
        
    def _load_model(self):
        """延迟加载OCR模型"""
        if self.reader is None:
            if self.backend == "easyocr":
                try:
                    import easyocr
                    self.reader = easyocr.Reader(self.languages, gpu=True)
                except ImportError:
                    raise ImportError("Please install easyocr: pip install easyocr")
                    
            elif self.backend == "paddleocr":
                try:
                    from paddleocr import PaddleOCR
                    self.reader = PaddleOCR(use_angle_cls=True, lang='ch' if 'ch' in self.languages else 'en')
                except ImportError:
                    raise ImportError("Please install paddleocr: pip install paddleocr")
                    
            elif self.backend == "tesseract":
                try:
                    import pytesseract
                    self.reader = pytesseract
                except ImportError:
                    raise ImportError("Please install pytesseract: pip install pytesseract")
    
    def process(self, image: Image.Image, bbox: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        提取图像中的文本
        
        Args:
            image: PIL图像
            bbox: 可选的感兴趣区域
        
        Returns:
            包含OCR结果的字典
        """
        self._load_model()
        
        # 如果指定了区域，先裁剪
        if bbox and len(bbox) == 4:
            image = image.crop(bbox)
        
        # 转换为numpy数组
        image_np = np.array(image)
        
        results = {
            "status": "success",
            "text_regions": [],
            "full_text": ""
        }
        
        if self.backend == "easyocr":
            detections = self.reader.readtext(image_np)
            
            for (bbox, text, prob) in detections:
                # EasyOCR返回的bbox是四个角点
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                results["text_regions"].append({
                    "bbox": [min(x_coords), min(y_coords), max(x_coords), max(y_coords)],
                    "text": text,
                    "confidence": float(prob)
                })
            
            results["full_text"] = " ".join([r["text"] for r in results["text_regions"]])
            
        elif self.backend == "paddleocr":
            result = self.reader.ocr(image_np, cls=True)
            
            for line in result:
                for box, (text, prob) in line:
                    results["text_regions"].append({
                        "bbox": [box[0][0], box[0][1], box[2][0], box[2][1]],
                        "text": text,
                        "confidence": float(prob)
                    })
            
            results["full_text"] = " ".join([r["text"] for r in results["text_regions"]])
            
        elif self.backend == "tesseract":
            text = self.reader.image_to_string(image)
            results["full_text"] = text.strip()
            
            # 获取详细信息
            data = self.reader.image_to_data(image, output_type=self.reader.Output.DICT)
            
            for i in range(len(data['text'])):
                if data['text'][i].strip():
                    results["text_regions"].append({
                        "bbox": [data['left'][i], data['top'][i], 
                                data['left'][i] + data['width'][i], 
                                data['top'][i] + data['height'][i]],
                        "text": data['text'][i],
                        "confidence": float(data['conf'][i]) / 100.0
                    })
        
        return results