# vlm_gym/tools/__init__.py

"""
Vision Tools for VLM Agent

This module provides various computer vision tools for the VLM Agent:
- Object detection and segmentation (SAM)
- Text extraction (OCR)
- Image processing (crop, zoom, annotate)
"""

from .base import BaseVisionTool
from .tool_manager import VisionToolManager

# Optional imports with graceful failure
try:
    from .sam_detector import SAMDetector
except ImportError:
    SAMDetector = None
    print("Warning: SAM detector not available. Install segment-anything to enable.")

try:
    from .ocr_tool import OCRTool
except ImportError:
    OCRTool = None
    print("Warning: OCR tool not available. Install easyocr/paddleocr/pytesseract to enable.")

try:
    from .image_processor import ImageProcessor
except ImportError:
    ImageProcessor = None
    print("Warning: Image processor not available.")

__all__ = [
    'BaseVisionTool',
    'VisionToolManager',
    'SAMDetector',
    'OCRTool', 
    'ImageProcessor'
]

# Version info
__version__ = '0.1.0'