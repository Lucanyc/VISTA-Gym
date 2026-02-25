# vlm_gym/environments/tools/grounding_dino.py
import sys
sys.path.append('/workspace/GroundingDINO')

import os
import torch
import torchvision
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Tuple, Optional

# Add GroundingDINO to Python path
#sys.path.append("/data/wang/meng/GYM-Work/vlm_gym-tool-usage-new/GroundingDINO")

from groundingdino.util.inference import load_model, predict
from groundingdino.util import box_ops
import groundingdino.datasets.transforms as T

from .base import ToolBase


class GroundingDINOTool(ToolBase):
    name = "grounding_dino"
    """
    Grounding DINO Tool - Open-vocabulary Object Detection
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        # Set tool attributes first
        self.name = "grounding_dino"
        self.description = "Open-vocabulary object detection tool that can detect objects described by arbitrary text"
        self.capabilities = ["Object Detection", "Localization", "Counting", "Open Vocabulary"]
        
        # ⭐ Fix: pass name parameter to parent class
        super().__init__(name=self.name)
        
        # Configuration
        self.config = config or {}
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = self.config.get('model_path')
        self.model_config = self.config.get('model_config')
        self.default_box_threshold = self.config.get('box_threshold', 0.35)
        self.default_text_threshold = self.config.get('text_threshold', 0.25)
        self.nms_threshold = self.config.get('nms_threshold', 0.8)
        
        # Lazy model loading
        self.model = None
        self.transform = None
        self.current_image = None
        self.current_image_np = None
        self.current_image_tensor = None
        
    def _load_model(self):
        """Lazy model loading"""
        if self.model is None:
            print(f"Loading Grounding DINO model from {self.model_path}")
            self.model = load_model(
                model_config_path=self.model_config,
                model_checkpoint_path=self.model_path,
                device=self.device
            )
            self.model.to(self.device)
            self.model.eval()
            
            # Initialize image transforms
            self.transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            
            print("Grounding DINO model loaded successfully")
    
    def reset(self, image: Image.Image):
        """Reset tool state to prepare for processing a new image"""
        self._load_model()  # Ensure model is loaded
        
        # Save original image
        self.current_image = image.convert("RGB")
        
        # Prepare numpy format
        self.current_image_np = np.asarray(self.current_image)
        
        # Prepare tensor format
        self.current_image_tensor, _ = self.transform(self.current_image, None)
        
    def execute(self, action_string: str) -> Dict[str, Any]:
        """Execute Grounding DINO detection"""
        import json
        
        if self.current_image is None:
            return {"error": "No image loaded. Please call reset() first."}
        
        # Parse parameters
        if isinstance(action_string, str):
            try:
                params = json.loads(action_string)
            except json.JSONDecodeError:
                # If not JSON, try using directly as caption
                params = {"caption": action_string}
        else:
            params = action_string
            
        # Get parameters
        caption = params.get("caption", "")
        if not caption:
            return {"error": "Caption is required for detection"}
            
        box_threshold = params.get("box_threshold", self.default_box_threshold)
        text_threshold = params.get("text_threshold", self.default_text_threshold)
        
        try:
            # Run prediction
            boxes, logits, phrases = predict(
                model=self.model,
                image=self.current_image_tensor,
                caption=caption,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=self.device
            )
            
            # Apply NMS
            boxes, logits, phrases = self._nms(boxes, logits, phrases)
            
            # Convert bounding box format: cxcywh -> xyxy
            boxes = box_ops.box_cxcywh_to_xyxy(boxes)
            
            # Convert to list format
            boxes = boxes.tolist()
            # Round to 2 decimal places
            boxes = [[round(x, 2) for x in box] for box in boxes]
            logits = logits.tolist()
            logits = [round(x, 2) for x in logits]
            
            # Get image dimensions
            h, w = self.current_image_np.shape[:2]
            
            result = {
                "boxes": boxes,
                "logits": logits,
                "phrases": phrases,
                "size": [h, w],  # H, W
                "num_detections": len(boxes)
            }
            
            return result
            
        except Exception as e:
            return {
                "error": f"Detection failed: {str(e)}",
                "error_type": type(e).__name__
            }
    
    def _nms(self, boxes, logits, phrases):
        """Apply Non-Maximum Suppression (NMS)"""
        # Convert to xyxy format for NMS
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes)
        
        # Execute NMS
        nms_idx = torchvision.ops.nms(boxes_xyxy, logits, self.nms_threshold)
        
        # Apply NMS indices
        boxes = boxes[nms_idx]
        logits = logits[nms_idx]
        phrases = [phrases[idx] for idx in nms_idx]
        
        return boxes, logits, phrases
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return tool capability description"""
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "parameters": {
                "caption": {
                    "type": "string",
                    "required": True,
                    "description": "Description of objects to detect, supports complex descriptions like 'red car and blue bicycle'"
                },
                "box_threshold": {
                    "type": "float",
                    "default": self.default_box_threshold,
                    "range": [0.0, 1.0],
                    "description": "Bounding box confidence threshold"
                },
                "text_threshold": {
                    "type": "float",
                    "default": self.default_text_threshold,
                    "range": [0.0, 1.0],
                    "description": "Text matching confidence threshold"
                }
            },
            "output": {
                "boxes": {
                    "type": "list",
                    "description": "List of bounding box coordinates in format [[x1,y1,x2,y2], ...]"
                },
                "logits": {
                    "type": "list",
                    "description": "Confidence score for each detected bounding box"
                },
                "phrases": {
                    "type": "list",
                    "description": "Text label corresponding to each bounding box"
                },
                "size": {
                    "type": "list",
                    "description": "Image dimensions [height, width]"
                },
                "num_detections": {
                    "type": "int",
                    "description": "Number of detected objects"
                }
            }
        }
