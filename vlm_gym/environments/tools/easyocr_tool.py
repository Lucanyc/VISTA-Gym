# easyocr_tool.py
import sys
sys.path.append('/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista')
sys.path.append('/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/vlm_gym/environments/tools/EasyOCR')

import os
import json
import re
import easyocr
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Tuple, Optional

from vlm_gym.environments.tools.base import ToolBase


class EasyOCRTool(ToolBase):
    name = "easyocr"
    """
    EasyOCR工具 - 文本检测和识别，支持多语言和数学符号
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        # 先设置工具属性
        self.name = "easyocr"
        self.description = "文本检测和识别工具，支持多语言和数学符号"
        self.capabilities = ["文本检测", "文本识别", "多语言支持", "数学符号识别", "表格文本提取"]
        
        # 传递name参数给父类
        super().__init__(name=self.name)

        # 配置
        self.config = config or {}
        
        # 默认语言列表（英文 + 简体中文）
        self.languages = self.config.get('languages', ['en', 'ch_sim'])
        
        # 置信度阈值
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        
        # 是否使用GPU
        self.gpu = self.config.get('gpu', False)
        
        # 延迟加载reader
        self.reader = None
        self.current_image = None
        self.current_image_path = None
        
        # 数学符号映射（用于后处理）
        self.math_corrections = {
            '410': '41°',
            '590': '59°',
            '900': '90°',
            '1800': '180°',
            '3600': '360°',
        }
        
        print(f"[EasyOCR] Initialized with languages: {self.languages}")
        print(f"[EasyOCR] GPU: {self.gpu}, Confidence threshold: {self.confidence_threshold}")
    
    def _load_reader(self):
        """延迟加载EasyOCR reader"""
        if self.reader is None:
            print(f"[EasyOCR] Loading reader with languages: {self.languages}")
            self.reader = easyocr.Reader(self.languages, gpu=self.gpu)
            print("[EasyOCR] Reader loaded successfully")
    
    def reset(self, image: Image.Image):
        """重置工具状态，准备处理新图像"""
        self._load_reader()  # 确保reader已加载
        
        # 处理输入 - 支持字符串路径或PIL Image
        if isinstance(image, str):
            self.current_image_path = image
            self.current_image = Image.open(image)
        else:
            self.current_image = image
            self.current_image_path = None
        
        # 确保图像是RGB格式
        if self.current_image.mode != 'RGB':
            self.current_image = self.current_image.convert('RGB')
        
        print(f"[EasyOCR] Reset with image size: {self.current_image.size}")
    
    def execute(self, action_string: str) -> Dict[str, Any]:
        """执行OCR任务"""
        
        if self.current_image is None:
            return {"error": "No image loaded. Please call reset() first."}
        
        # 解析参数
        if isinstance(action_string, str):
            try:
                params = json.loads(action_string)
            except json.JSONDecodeError:
                # 如果不是JSON，尝试作为任务类型直接使用
                params = {"task": action_string}
        else:
            params = action_string
        
        # 获取任务类型
        task = params.get("task", "detect_and_recognize")
        
        print(f"[EasyOCR] Executing task: {task}")
        
        try:
            if task == "detect_and_recognize":
                return self._detect_and_recognize(params)
            elif task == "detect_only":
                return self._detect_only(params)
            elif task == "recognize_only":
                return self._recognize_only(params)
            elif task == "extract_math":
                return self._extract_math(params)
            elif task == "extract_table":
                return self._extract_table(params)
            elif task == "extract_by_region":
                return self._extract_by_region(params)
            else:
                return {
                    "error": f"Unknown task: {task}",
                    "available_tasks": [
                        "detect_and_recognize",
                        "detect_only", 
                        "recognize_only",
                        "extract_math",
                        "extract_table",
                        "extract_by_region"
                    ]
                }
        except Exception as e:
            import traceback
            return {
                "error": f"OCR execution failed: {str(e)}",
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()
            }
    
    def _detect_and_recognize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """检测并识别文本"""
        # 执行OCR
        if self.current_image_path:
            results = self.reader.readtext(self.current_image_path)
        else:
            # 转换为numpy数组
            img_array = np.array(self.current_image)
            results = self.reader.readtext(img_array)
        
        # 过滤低置信度结果
        min_confidence = params.get('min_confidence', self.confidence_threshold)
        filtered_results = []
        
        for bbox, text, confidence in results:
            if confidence >= min_confidence:
                # 应用数学符号修正
                corrected_text = self._correct_math_symbols(text)
                filtered_results.append({
                    "bbox": bbox,
                    "text": corrected_text,
                    "original_text": text if corrected_text != text else None,
                    "confidence": float(confidence)
                })
        
        # 按位置排序（从上到下，从左到右）
        filtered_results.sort(key=lambda x: (min(p[1] for p in x['bbox']), min(p[0] for p in x['bbox'])))
        
        # 提取所有文本
        all_texts = [r['text'] for r in filtered_results]
        
        return {
            "success": True,
            "task": "detect_and_recognize",
            "num_detections": len(filtered_results),
            "detections": filtered_results,
            "all_texts": all_texts,
            "processed_output": " ".join(all_texts)
        }
    
    def _detect_only(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """仅检测文本区域，不识别内容"""
        # 这需要访问检测网络
        # 简化版：使用readtext但忽略文本内容
        results = self.reader.readtext(np.array(self.current_image))
        
        regions = []
        for bbox, _, confidence in results:
            if confidence >= self.confidence_threshold:
                regions.append({
                    "bbox": bbox,
                    "confidence": float(confidence),
                    "area": self._calculate_bbox_area(bbox)
                })
        
        return {
            "success": True,
            "task": "detect_only",
            "num_regions": len(regions),
            "regions": regions
        }
    
    def _recognize_only(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """仅识别文本内容（简化版）"""
        results = self.reader.readtext(np.array(self.current_image), detail=0)
        
        # 应用数学符号修正
        corrected_texts = [self._correct_math_symbols(text) for text in results]
        
        return {
            "success": True,
            "task": "recognize_only",
            "texts": corrected_texts,
            "processed_output": " ".join(corrected_texts)
        }
    
    def _extract_math(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """提取数学相关内容"""
        results = self.reader.readtext(np.array(self.current_image))
        
        math_elements = {
            "numbers": [],
            "symbols": [],
            "angles": [],
            "variables": [],
            "expressions": []
        }
        
        # 数学符号模式
        math_symbols = ['=', '+', '-', '×', '÷', '∠', '°', 'π', '∞', '∑', '√', '∫', '±', '≈', '≠', '≤', '≥']
        angle_pattern = r'\d+\s*°|\d+°|∠\s*\w+'
        variable_pattern = r'^[a-zA-Z]$|^[a-zA-Z]\d*$'
        
        for bbox, text, confidence in results:
            if confidence < self.confidence_threshold:
                continue
            
            # 修正文本
            text = self._correct_math_symbols(text)
            
            # 分类
            if re.match(r'^-?\d+\.?\d*$', text):  # 纯数字
                math_elements["numbers"].append({
                    "value": text,
                    "bbox": bbox,
                    "confidence": float(confidence)
                })
            elif re.search(angle_pattern, text):  # 角度
                math_elements["angles"].append({
                    "value": text,
                    "bbox": bbox,
                    "confidence": float(confidence)
                })
            elif any(symbol in text for symbol in math_symbols):  # 包含数学符号
                math_elements["symbols"].append({
                    "value": text,
                    "bbox": bbox,
                    "confidence": float(confidence)
                })
            elif re.match(variable_pattern, text):  # 变量
                math_elements["variables"].append({
                    "value": text,
                    "bbox": bbox,
                    "confidence": float(confidence)
                })
            else:  # 可能是表达式
                math_elements["expressions"].append({
                    "value": text,
                    "bbox": bbox,
                    "confidence": float(confidence)
                })
        
        # 尝试组合相邻的数学元素
        combined_expressions = self._combine_math_elements(math_elements)
        
        return {
            "success": True,
            "task": "extract_math",
            "math_elements": math_elements,
            "combined_expressions": combined_expressions,
            "summary": {
                "num_numbers": len(math_elements["numbers"]),
                "num_symbols": len(math_elements["symbols"]),
                "num_angles": len(math_elements["angles"]),
                "num_variables": len(math_elements["variables"]),
                "num_expressions": len(math_elements["expressions"])
            }
        }
    
    def _extract_table(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """提取表格文本（基于位置关系）"""
        results = self.reader.readtext(np.array(self.current_image))
        
        if not results:
            return {
                "success": True,
                "task": "extract_table",
                "table_data": [],
                "processed_output": "No text detected"
            }
        
        # 根据y坐标分组（行）
        rows = {}
        tolerance = params.get('row_tolerance', 10)  # y坐标容差
        
        for bbox, text, confidence in results:
            if confidence < self.confidence_threshold:
                continue
            
            # 计算中心y坐标
            center_y = sum(p[1] for p in bbox) / len(bbox)
            
            # 找到对应的行
            row_found = False
            for row_y in rows:
                if abs(row_y - center_y) < tolerance:
                    rows[row_y].append({
                        'x': min(p[0] for p in bbox),
                        'text': self._correct_math_symbols(text),
                        'bbox': bbox
                    })
                    row_found = True
                    break
            
            if not row_found:
                rows[center_y] = [{
                    'x': min(p[0] for p in bbox),
                    'text': self._correct_math_symbols(text),
                    'bbox': bbox
                }]
        
        # 排序行并在每行内排序列
        sorted_rows = []
        for row_y in sorted(rows.keys()):
            row_items = sorted(rows[row_y], key=lambda x: x['x'])
            row_texts = [item['text'] for item in row_items]
            sorted_rows.append(row_texts)
        
        # 格式化为表格字符串
        table_str = ""
        for row in sorted_rows:
            table_str += " | ".join(row) + "\n"
        
        return {
            "success": True,
            "task": "extract_table",
            "table_data": sorted_rows,
            "processed_output": table_str.strip()
        }
    
    def _extract_by_region(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """提取指定区域的文本"""
        region = params.get("region", None)
        if not region:
            return {"error": "Region parameter is required"}
        
        # region格式：[x1, y1, x2, y2]
        x1, y1, x2, y2 = region
        
        # 裁剪图像
        cropped = self.current_image.crop((x1, y1, x2, y2))
        
        # 在裁剪区域执行OCR
        results = self.reader.readtext(np.array(cropped))
        
        # 调整坐标到原图
        adjusted_results = []
        for bbox, text, confidence in results:
            if confidence >= self.confidence_threshold:
                # 调整bbox坐标
                adjusted_bbox = [[p[0] + x1, p[1] + y1] for p in bbox]
                adjusted_results.append({
                    "bbox": adjusted_bbox,
                    "text": self._correct_math_symbols(text),
                    "confidence": float(confidence)
                })
        
        return {
            "success": True,
            "task": "extract_by_region",
            "region": region,
            "detections": adjusted_results,
            "processed_output": " ".join([r['text'] for r in adjusted_results])
        }
    
    def _correct_math_symbols(self, text: str) -> str:
        """修正常见的数学符号识别错误"""
        # 应用预定义的修正
        for wrong, correct in self.math_corrections.items():
            if text == wrong:
                return correct
        
        # 其他修正规则
        # 修正角度（如 "41 0" -> "41°"）
        text = re.sub(r'(\d+)\s*0$', r'\1°', text)
        
        # 修正常见的OCR错误
        text = text.replace('。', '°')  # 中文句号误识别为度数符号
        
        return text
    
    def _calculate_bbox_area(self, bbox: List[List[float]]) -> float:
        """计算边界框面积"""
        # 使用鞋带公式计算多边形面积
        n = len(bbox)
        area = 0
        for i in range(n):
            j = (i + 1) % n
            area += bbox[i][0] * bbox[j][1]
            area -= bbox[j][0] * bbox[i][1]
        return abs(area) / 2
    
    def _combine_math_elements(self, math_elements: Dict) -> List[str]:
        """尝试组合相邻的数学元素形成完整表达式"""
        # 简化版：基于位置关系组合
        all_elements = []
        
        # 收集所有元素及其位置
        for category, items in math_elements.items():
            for item in items:
                center_x = sum(p[0] for p in item['bbox']) / len(item['bbox'])
                center_y = sum(p[1] for p in item['bbox']) / len(item['bbox'])
                all_elements.append({
                    'text': item['value'],
                    'x': center_x,
                    'y': center_y,
                    'bbox': item['bbox']
                })
        
        # 按位置排序
        all_elements.sort(key=lambda e: (e['y'], e['x']))
        
        # 简单组合：同一行的元素
        combined = []
        if all_elements:
            current_line = [all_elements[0]['text']]
            current_y = all_elements[0]['y']
            
            for elem in all_elements[1:]:
                if abs(elem['y'] - current_y) < 20:  # 同一行
                    current_line.append(elem['text'])
                else:
                    combined.append(' '.join(current_line))
                    current_line = [elem['text']]
                    current_y = elem['y']
            
            if current_line:
                combined.append(' '.join(current_line))
        
        return combined
    
    def get_capabilities(self) -> Dict[str, Any]:
        """返回工具能力描述"""
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "languages": self.languages,
            "parameters": {
                "task": {
                    "type": "string",
                    "required": True,
                    "enum": [
                        "detect_and_recognize",
                        "detect_only",
                        "recognize_only", 
                        "extract_math",
                        "extract_table",
                        "extract_by_region"
                    ],
                    "description": "OCR任务类型"
                },
                "min_confidence": {
                    "type": "float",
                    "default": self.confidence_threshold,
                    "range": [0.0, 1.0],
                    "description": "最小置信度阈值"
                },
                "region": {
                    "type": "array",
                    "required": False,
                    "description": "指定区域[x1, y1, x2, y2]（仅extract_by_region任务需要）"
                },
                "row_tolerance": {
                    "type": "integer", 
                    "default": 10,
                    "description": "表格行检测的y坐标容差（仅extract_table任务）"
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
                "detections": {
                    "type": "array",
                    "description": "检测结果列表"
                },
                "processed_output": {
                    "type": "string",
                    "description": "处理后的文本输出"
                }
            }
        }