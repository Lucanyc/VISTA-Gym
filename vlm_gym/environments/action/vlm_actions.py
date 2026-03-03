# vlm_gym/environments/action/vlm_actions.py
"""
VLM特定的图像处理动作
包含基础VLM动作和Chain-of-Focus工具
"""
from typing import Dict, List, Any, Optional, Union
import numpy as np
from PIL import Image
from pathlib import Path
import json

# ==================== 原有的基础VLM动作 ====================

def analyze_image(image_path: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
    """
    分析图像内容并提取信息
    
    Examples:
        analyze_image("/path/to/image.jpg", "objects")
        analyze_image("/path/to/image.jpg", "scene")
        analyze_image("/path/to/image.jpg", "text")
    """
    valid_types = ["objects", "scene", "text", "comprehensive"]
    if analysis_type not in valid_types:
        raise ValueError(f"Invalid analysis type. Must be one of {valid_types}")
    
    # TODO: 调用VLM模型
    result = {
        "image_path": image_path,
        "analysis_type": analysis_type,
        "results": {}
    }
    
    if analysis_type in ["objects", "comprehensive"]:
        result["results"]["objects"] = [
            {"label": "example_object", "confidence": 0.95}
        ]
    
    return result

def detect_objects(image_path: str, confidence_threshold: float = 0.5) -> List[Dict]:
    """
    检测图像中的对象
    
    Examples:
        detect_objects("/path/to/image.jpg")
        detect_objects("/path/to/image.jpg", confidence_threshold=0.7)
    """
    if confidence_threshold < 0 or confidence_threshold > 1:
        raise ValueError("Confidence threshold must be between 0 and 1")
    
    # TODO: 调用目标检测模型
    detected_objects = [
        {"label": "object1", "confidence": 0.95, "bbox": [10, 10, 100, 100]}
    ]
    
    return [obj for obj in detected_objects if obj["confidence"] >= confidence_threshold]

def extract_text(image_path: str, region: Optional[List[int]] = None) -> str:
    """
    从图像中提取文本（OCR）
    
    Examples:
        extract_text("/path/to/image.jpg")
        extract_text("/path/to/image.jpg", region=[100, 100, 200, 200])
    """
    if region and len(region) != 4:
        raise ValueError("Region must have 4 coordinates: [x1, y1, x2, y2]")
    
    # TODO: 调用OCR模型
    if region:
        return f"Text from region {region}"
    return "Extracted text from image"

def describe_region(image_path: str, bbox: List[int]) -> str:
    """
    描述图像中的特定区域
    
    Examples:
        describe_region("/path/to/image.jpg", [100, 100, 200, 200])
    """
    if len(bbox) != 4:
        raise ValueError("Bbox must have 4 coordinates")
    
    # TODO: 调用VLM模型
    return f"Description of region {bbox}"

def compare_images(image_paths: List[str], aspect: str = "similarity") -> Dict[str, Any]:
    """
    比较多张图像
    
    Examples:
        compare_images(["/path/img1.jpg", "/path/img2.jpg"], "similarity")
        compare_images(["/path/img1.jpg", "/path/img2.jpg"], "differences")
    """
    if len(image_paths) < 2:
        raise ValueError("Need at least 2 images")
    
    valid_aspects = ["similarity", "differences", "content"]
    if aspect not in valid_aspects:
        raise ValueError(f"Invalid aspect. Must be one of {valid_aspects}")
    
    # TODO: 调用比较模型
    return {
        "images": image_paths,
        "aspect": aspect,
        "result": f"Comparison based on {aspect}"
    }

def visual_qa(image_path: str, question: str, context: Optional[Dict] = None) -> str:
    """
    视觉问答
    
    Examples:
        visual_qa("/path/to/image.jpg", "What color is the car?")
        visual_qa("/path/to/image.jpg", "How many people?", {"focus": "foreground"})
    """
    if not question:
        raise ValueError("Question cannot be empty")
    
    # TODO: 调用VQA模型
    answer = f"Answer to '{question}'"
    if context:
        answer += f" (with context: {context})"
    
    return answer

def generate_caption(image_path: str, style: str = "descriptive") -> str:
    """
    生成图像描述
    
    Examples:
        generate_caption("/path/to/image.jpg")
        generate_caption("/path/to/image.jpg", style="brief")
    """
    valid_styles = ["brief", "descriptive", "detailed"]
    if style not in valid_styles:
        raise ValueError(f"Invalid style. Must be one of {valid_styles}")
    
    # TODO: 调用描述生成模型
    return f"Image caption in {style} style"

def segment_image(image_path: str, mode: str = "semantic") -> np.ndarray:
    """
    图像分割
    
    Examples:
        segment_image("/path/to/image.jpg", mode="semantic")
        segment_image("/path/to/image.jpg", mode="instance")
    """
    valid_modes = ["semantic", "instance", "panoptic"]
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode. Must be one of {valid_modes}")
    
    # TODO: 调用分割模型
    return np.zeros((224, 224), dtype=np.uint8)

def visual_grounding(image_path: str, description: str) -> List[int]:
    """
    根据描述定位图像区域
    
    Examples:
        visual_grounding("/path/to/image.jpg", "the red car")
    """
    if not description:
        raise ValueError("Description cannot be empty")
    
    # TODO: 调用定位模型
    return [100, 100, 200, 200]


# ==================== Chain-of-Focus 工具 ====================

def locate_object(image_path: str, target_object: str, region_of_interest: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    定位图像中的特定对象并返回边界框
    基于论文中的Locate Tool - 支持粗到细的定位
    
    Examples:
        locate_object("/path/to/image.jpg", "red car")
        locate_object("/path/to/image.jpg", "person wearing hat", [100, 100, 500, 500])
        locate_object("/path/to/image.jpg", "text on the sign")
        locate_object("/path/to/image.jpg", "dog's head", [200, 100, 400, 300])
    """
    # 验证输入
    if not target_object:
        raise ValueError("Target object description cannot be empty")
    
    if region_of_interest and len(region_of_interest) != 4:
        raise ValueError("Region of interest must have 4 coordinates: [x1, y1, x2, y2]")
    
    # TODO: 调用对象检测模型（如Grounding DINO、OWL-ViT、SAM等）
    # 实际实现时应该：
    # 1. 加载图像
    # 2. 如果有ROI，裁剪到该区域
    # 3. 使用检测模型查找目标对象
    # 4. 返回边界框坐标
    
    result = {
        "image_path": image_path,
        "target_object": target_object,
        "status": "SUCCESS",
        "detections": []
    }
    
    if region_of_interest:
        result["search_region"] = region_of_interest
        # 模拟在ROI内的检测结果
        x1, y1, x2, y2 = region_of_interest
        detection = {
            "bbox": [x1 + 50, y1 + 50, x2 - 50, y2 - 50],  # 在ROI内的示例边界框
            "confidence": 0.92,
            "label": target_object
        }
    else:
        # 模拟全图检测结果
        detection = {
            "bbox": [231, 174, 1023, 482],  # [x1, y1, x2, y2]
            "confidence": 0.95,
            "label": target_object
        }
    
    result["detections"].append(detection)
    
    # 如果没有检测到对象
    if not result["detections"]:
        result["status"] = "NOT_FOUND"
        result["message"] = f"Could not locate '{target_object}' in the image"
    
    return result

def vlm_understand(image_path: str, question: str, bbox: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    使用VLM理解图像内容并回答问题
    基于论文中的VLM Understanding Tool - 支持全图或特定区域分析
    
    Examples:
        vlm_understand("/path/to/image.jpg", "What is the color of the car?")
        vlm_understand("/path/to/image.jpg", "What text is on the sign?", [100, 100, 200, 200])
        vlm_understand("/path/to/image.jpg", "How many people are in this region?", [0, 0, 500, 500])
        vlm_understand("/path/to/image.jpg", "What brand is on the bottle?", [150, 200, 250, 400])
    """
    if not question:
        raise ValueError("Question cannot be empty")
    
    if bbox and len(bbox) != 4:
        raise ValueError("Bbox must have 4 coordinates: [x1, y1, x2, y2]")
    
    # TODO: 调用VLM模型（如Qwen-VL-Max、GPT-4V、Claude等）
    # 实际实现时应该：
    # 1. 加载图像
    # 2. 如果有bbox，裁剪到该区域
    # 3. 构建prompt：图像 + 问题
    # 4. 调用VLM获取答案
    
    result = {
        "image_path": image_path,
        "question": question,
        "status": "SUCCESS",
        "answer": "",
        "reasoning": "",
        "confidence": 0.0
    }
    
    if bbox:
        # 分析指定区域
        result["analyzed_region"] = bbox
        result["answer"] = f"Based on the region {bbox}, the answer is: [VLM response]"
        result["reasoning"] = f"Focused analysis on bbox {bbox} shows specific details about {question}"
        result["confidence"] = 0.85
    else:
        # 分析整个图像
        result["answer"] = f"Analyzing the entire image: [VLM response to {question}]"
        result["reasoning"] = "Performed comprehensive analysis of the full image"
        result["confidence"] = 0.90
    
    # 模拟一些具体的回答示例
    if "color" in question.lower():
        result["answer"] = "The color appears to be red"
    elif "text" in question.lower() or "number" in question.lower():
        result["answer"] = "The text shows: ABC123"
    elif "how many" in question.lower():
        result["answer"] = "There are 3 items visible"
    
    return result

def adjust_bbox(image_path: str, current_bbox: List[int], instruction: str) -> Dict[str, Any]:
    """
    根据指令调整边界框
    基于论文中的Adjust BBox Tool - 支持灵活的边界框调整
    
    Examples:
        adjust_bbox("/path/to/image.jpg", [100, 100, 200, 200], "expand left by 50 pixels")
        adjust_bbox("/path/to/image.jpg", [100, 100, 200, 200], "shrink top edge by 20%")
        adjust_bbox("/path/to/image.jpg", [100, 100, 200, 200], "focus on the upper half")
        adjust_bbox("/path/to/image.jpg", [100, 100, 200, 200], "zoom out to include surroundings")
        adjust_bbox("/path/to/image.jpg", [100, 100, 200, 200], "shift right by 100 pixels")
    """
    if len(current_bbox) != 4:
        raise ValueError("Current bbox must have 4 coordinates: [x1, y1, x2, y2]")
    
    if not instruction:
        raise ValueError("Adjustment instruction cannot be empty")
    
    # 获取当前边界框坐标
    x1, y1, x2, y2 = current_bbox
    adjusted_bbox = current_bbox.copy()
    
    # 解析指令并调整边界框
    instruction_lower = instruction.lower()
    
    # 扩展操作
    if "expand" in instruction_lower or "enlarge" in instruction_lower:
        pixels = 50  # 默认扩展像素
        percent = 0.1  # 默认扩展百分比
        
        # 提取数值
        import re
        pixel_match = re.search(r'(\d+)\s*pixel', instruction_lower)
        percent_match = re.search(r'(\d+)\s*%', instruction_lower)
        
        if pixel_match:
            pixels = int(pixel_match.group(1))
        if percent_match:
            percent = int(percent_match.group(1)) / 100.0
        
        width = x2 - x1
        height = y2 - y1
        
        if "left" in instruction_lower:
            adjusted_bbox[0] -= pixels if pixel_match else int(width * percent)
        if "right" in instruction_lower:
            adjusted_bbox[2] += pixels if pixel_match else int(width * percent)
        if "top" in instruction_lower or "up" in instruction_lower:
            adjusted_bbox[1] -= pixels if pixel_match else int(height * percent)
        if "bottom" in instruction_lower or "down" in instruction_lower:
            adjusted_bbox[3] += pixels if pixel_match else int(height * percent)
        if not any(direction in instruction_lower for direction in ["left", "right", "top", "bottom", "up", "down"]):
            # 四周均匀扩展
            expansion = pixels if pixel_match else int(min(width, height) * percent)
            adjusted_bbox[0] -= expansion
            adjusted_bbox[1] -= expansion
            adjusted_bbox[2] += expansion
            adjusted_bbox[3] += expansion
    
    # 缩小操作
    elif "shrink" in instruction_lower or "narrow" in instruction_lower:
        pixels = 20  # 默认缩小像素
        percent = 0.1  # 默认缩小百分比
        
        pixel_match = re.search(r'(\d+)\s*pixel', instruction_lower)
        percent_match = re.search(r'(\d+)\s*%', instruction_lower)
        
        if pixel_match:
            pixels = int(pixel_match.group(1))
        if percent_match:
            percent = int(percent_match.group(1)) / 100.0
        
        width = x2 - x1
        height = y2 - y1
        
        if "left" in instruction_lower:
            adjusted_bbox[0] += pixels if pixel_match else int(width * percent)
        if "right" in instruction_lower:
            adjusted_bbox[2] -= pixels if pixel_match else int(width * percent)
        if "top" in instruction_lower:
            adjusted_bbox[1] += pixels if pixel_match else int(height * percent)
        if "bottom" in instruction_lower:
            adjusted_bbox[3] -= pixels if pixel_match else int(height * percent)
    
    # 聚焦操作
    elif "focus" in instruction_lower:
        if "upper half" in instruction_lower:
            mid_y = (y1 + y2) // 2
            adjusted_bbox[3] = mid_y
        elif "lower half" in instruction_lower:
            mid_y = (y1 + y2) // 2
            adjusted_bbox[1] = mid_y
        elif "left half" in instruction_lower:
            mid_x = (x1 + x2) // 2
            adjusted_bbox[2] = mid_x
        elif "right half" in instruction_lower:
            mid_x = (x1 + x2) // 2
            adjusted_bbox[0] = mid_x
        elif "center" in instruction_lower:
            width = x2 - x1
            height = y2 - y1
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            adjusted_bbox = [
                center_x - width // 4,
                center_y - height // 4,
                center_x + width // 4,
                center_y + height // 4
            ]
    
    # 移动操作
    elif "shift" in instruction_lower or "move" in instruction_lower:
        pixels = 50  # 默认移动像素
        pixel_match = re.search(r'(\d+)\s*pixel', instruction_lower)
        if pixel_match:
            pixels = int(pixel_match.group(1))
        
        if "left" in instruction_lower:
            adjusted_bbox[0] -= pixels
            adjusted_bbox[2] -= pixels
        if "right" in instruction_lower:
            adjusted_bbox[0] += pixels
            adjusted_bbox[2] += pixels
        if "up" in instruction_lower:
            adjusted_bbox[1] -= pixels
            adjusted_bbox[3] -= pixels
        if "down" in instruction_lower:
            adjusted_bbox[1] += pixels
            adjusted_bbox[3] += pixels
    
    # 缩放操作
    elif "zoom" in instruction_lower:
        if "in" in instruction_lower:
            # 缩小边界框（放大内容）
            factor = 0.8
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            width = int((x2 - x1) * factor)
            height = int((y2 - y1) * factor)
            adjusted_bbox = [
                center_x - width // 2,
                center_y - height // 2,
                center_x + width // 2,
                center_y + height // 2
            ]
        elif "out" in instruction_lower:
            # 扩大边界框（缩小内容）
            factor = 1.2
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            width = int((x2 - x1) * factor)
            height = int((y2 - y1) * factor)
            adjusted_bbox = [
                center_x - width // 2,
                center_y - height // 2,
                center_x + width // 2,
                center_y + height // 2
            ]
    
    # 确保边界框有效（不超出图像边界的基本检查）
    adjusted_bbox[0] = max(0, adjusted_bbox[0])
    adjusted_bbox[1] = max(0, adjusted_bbox[1])
    # 注意：这里应该检查图像实际尺寸，但需要加载图像
    
    # 确保边界框不会变成无效的（宽度或高度为负）
    if adjusted_bbox[2] <= adjusted_bbox[0]:
        adjusted_bbox[2] = adjusted_bbox[0] + 10
    if adjusted_bbox[3] <= adjusted_bbox[1]:
        adjusted_bbox[3] = adjusted_bbox[1] + 10
    
    return {
        "image_path": image_path,
        "original_bbox": current_bbox,
        "adjusted_bbox": adjusted_bbox,
        "instruction": instruction,
        "applied_adjustment": f"Adjusted from {current_bbox} to {adjusted_bbox}",
        "status": "SUCCESS"
    }

def verify_answer(question: str, predicted_answer: str, ground_truth: Optional[str] = None, 
                  reasoning_process: Optional[str] = None, image_path: Optional[str] = None) -> Dict[str, Any]:
    """
    验证答案的正确性并提供反馈
    基于论文中的Verification Tool - 评估答案和推理过程
    
    Examples:
        verify_answer("What color is the car?", "red", "blue")
        verify_answer("How many people?", "3", "3")
        verify_answer("What's the sign text?", "STOP", reasoning_process="Located sign at [100,100,200,200]")
        verify_answer("What number is on the jersey?", "8", "8", "Found player and zoomed to jersey area")
    """
    result = {
        "question": question,
        "predicted_answer": predicted_answer,
        "verification_status": "PENDING",
        "feedback": "",
        "suggestions": [],
        "confidence": 0.0
    }
    
    if image_path:
        result["image_path"] = image_path
    
    # 如果提供了真实答案，进行比较
    if ground_truth is not None:
        result["ground_truth"] = ground_truth
        
        # 进行智能比较（考虑大小写、空格、同义词等）
        pred_normalized = predicted_answer.lower().strip()
        truth_normalized = ground_truth.lower().strip()
        
        # 完全匹配
        if pred_normalized == truth_normalized:
            result["verification_status"] = "CORRECT"
            result["feedback"] = "The answer is correct."
            result["confidence"] = 1.0
        
        # 部分匹配
        elif pred_normalized in truth_normalized or truth_normalized in pred_normalized:
            result["verification_status"] = "PARTIALLY_CORRECT"
            result["feedback"] = f"The answer is partially correct. Expected '{ground_truth}' but got '{predicted_answer}'."
            result["confidence"] = 0.5
            result["suggestions"] = [
                "Try to be more specific in your answer",
                "Check if you've captured all the details"
            ]
        
        # 数字答案的特殊处理
        elif any(char.isdigit() for char in predicted_answer) and any(char.isdigit() for char in ground_truth):
            import re
            pred_numbers = re.findall(r'\d+', predicted_answer)
            truth_numbers = re.findall(r'\d+', ground_truth)
            
            if pred_numbers and truth_numbers and pred_numbers[0] == truth_numbers[0]:
                result["verification_status"] = "CORRECT"
                result["feedback"] = "The numerical answer is correct."
                result["confidence"] = 0.9
            else:
                result["verification_status"] = "INCORRECT"
                result["feedback"] = f"The answer is incorrect. Expected '{ground_truth}' but got '{predicted_answer}'."
                result["confidence"] = 0.1
        
        # 完全不匹配
        else:
            result["verification_status"] = "INCORRECT"
            result["feedback"] = f"The answer is incorrect. Expected '{ground_truth}' but got '{predicted_answer}'."
            result["confidence"] = 0.0
            
            # 根据问题类型提供具体建议
            if "color" in question.lower():
                result["suggestions"] = [
                    "Try focusing on the object more clearly",
                    "Consider lighting conditions that might affect color perception",
                    "Zoom in on the specific object to see its true color"
                ]
            elif "number" in question.lower() or "count" in question.lower():
                result["suggestions"] = [
                    "Try counting more carefully",
                    "Zoom in to see all items clearly",
                    "Check if some objects are partially hidden"
                ]
            elif "text" in question.lower() or "sign" in question.lower():
                result["suggestions"] = [
                    "The text might not be clearly visible",
                    "Try zooming in on the text area",
                    "Adjust the bounding box to focus on the text region"
                ]
            else:
                result["suggestions"] = [
                    "Try focusing on a different region of the image",
                    "Consider zooming in for more details",
                    "Re-examine the question requirements"
                ]
    
    # 评估推理过程
    if reasoning_process:
        result["reasoning_evaluation"] = _evaluate_reasoning(reasoning_process)
        
        # 基于推理质量调整建议
        if result["reasoning_evaluation"]["has_visual_grounding"]:
            if result["verification_status"] == "INCORRECT":
                result["suggestions"].insert(0, "Your visual grounding seems correct, try adjusting the bounding box for better view")
        else:
            result["suggestions"].insert(0, "Try to ground your answer in specific image regions")
    
    return result

def _evaluate_reasoning(reasoning_process: str) -> Dict[str, Any]:
    """
    辅助函数：评估推理过程的质量
    """
    evaluation = {
        "is_logical": True,
        "has_visual_grounding": False,
        "uses_zooming": False,
        "step_count": 0,
        "grounding_quality": "none"
    }
    
    # 检查是否包含视觉定位信息
    grounding_keywords = ["bbox", "bounding box", "region", "located", "found at", "[", "]"]
    if any(keyword in reasoning_process.lower() for keyword in grounding_keywords):
        evaluation["has_visual_grounding"] = True
        
        # 评估定位质量
        if "zoom" in reasoning_process.lower():
            evaluation["uses_zooming"] = True
            evaluation["grounding_quality"] = "detailed"
        elif reasoning_process.count("[") >= 2:  # 多个边界框
            evaluation["grounding_quality"] = "iterative"
        else:
            evaluation["grounding_quality"] = "basic"
    
    # 计算推理步骤
    step_indicators = ["first", "then", "next", "finally", "step", "after"]
    evaluation["step_count"] = sum(1 for indicator in step_indicators if indicator in reasoning_process.lower())
    
    # 检查逻辑性
    if evaluation["step_count"] > 0 or evaluation["has_visual_grounding"]:
        evaluation["is_logical"] = True
    
    return evaluation

def visual_search_and_reason(image_path: str, question: str, max_attempts: int = 3, 
                            enable_zoom: bool = True) -> Dict[str, Any]:
    """
    执行完整的视觉搜索和推理流程
    结合定位、理解、调整和验证的综合工具
    
    Examples:
        visual_search_and_reason("/path/to/image.jpg", "What's the license plate number?")
        visual_search_and_reason("/path/to/image.jpg", "What brand is on the bottle?", max_attempts=5)
        visual_search_and_reason("/path/to/image.jpg", "How many cars are there?", enable_zoom=False)
    """
    if max_attempts < 1:
        raise ValueError("Max attempts must be at least 1")
    
    result = {
        "image_path": image_path,
        "question": question,
        "attempts": [],
        "final_answer": None,
        "success": False,
        "total_attempts": 0,
        "reasoning_chain": []
    }
    
    # TODO: 实现完整的搜索流程
    # 这里提供一个简化的流程框架
    
    for attempt in range(max_attempts):
        attempt_data = {
            "attempt_number": attempt + 1,
            "actions": [],
            "answer": None,
            "verified": False
        }
        
        # 步骤1：初次尝试理解整个图像
        if attempt == 0:
            understanding = vlm_understand(image_path, question)
            attempt_data["actions"].append({
                "action": "initial_understanding",
                "result": understanding
            })
            attempt_data["answer"] = understanding["answer"]
        
        # 步骤2：如果启用缩放且前次尝试失败，尝试定位相关对象
        elif enable_zoom:
            # 基于问题推断需要定位的对象
            if "number" in question.lower() or "text" in question.lower():
                target = "text or numbers"
            elif "color" in question.lower():
                target = "the main object"
            else:
                target = "relevant object for the question"
            
            location_result = locate_object(image_path, target)
            attempt_data["actions"].append({
                "action": "locate_object",
                "result": location_result
            })
            
            if location_result["status"] == "SUCCESS" and location_result["detections"]:
                bbox = location_result["detections"][0]["bbox"]
                
                # 步骤3：如果需要，调整边界框
                if attempt > 1:
                    adjust_result = adjust_bbox(image_path, bbox, "zoom in")
                    attempt_data["actions"].append({
                        "action": "adjust_bbox",
                        "result": adjust_result
                    })
                    bbox = adjust_result["adjusted_bbox"]
                
                # 步骤4：在新区域重新理解
                understanding = vlm_understand(image_path, question, bbox)
                attempt_data["actions"].append({
                    "action": "focused_understanding",
                    "result": understanding
                })
                attempt_data["answer"] = understanding["answer"]
        
        # 步骤5：验证答案（如果可能）
        if attempt_data["answer"]:
            # 这里可以添加自验证逻辑
            attempt_data["verified"] = True
            result["final_answer"] = attempt_data["answer"]
            result["success"] = True
        
        result["attempts"].append(attempt_data)
        result["total_attempts"] = attempt + 1
        
        # 如果成功，结束循环
        if result["success"]:
            break
    
    # 构建推理链
    for attempt in result["attempts"]:
        for action in attempt["actions"]:
            result["reasoning_chain"].append({
                "step": len(result["reasoning_chain"]) + 1,
                "action": action["action"],
                "outcome": action["result"].get("answer", action["result"].get("status", "completed"))
            })
    
    return result

def extract_and_focus(image_path: str, target_description: str, extraction_mode: str = "auto") -> Dict[str, Any]:
    """
    提取并聚焦于图像中的特定目标
    结合检测和裁剪功能的便捷工具
    
    Examples:
        extract_and_focus("/path/to/image.jpg", "the red car")
        extract_and_focus("/path/to/image.jpg", "text on the sign", "text")
        extract_and_focus("/path/to/image.jpg", "person's face", "object")
    """
    valid_modes = ["auto", "text", "object", "region"]
    if extraction_mode not in valid_modes:
        raise ValueError(f"Invalid extraction mode. Must be one of {valid_modes}")
    
    result = {
        "image_path": image_path,
        "target": target_description,
        "mode": extraction_mode,
        "extracted_regions": [],
        "status": "PENDING"
    }
    
    # 根据模式选择合适的提取方法
    if extraction_mode == "text" or (extraction_mode == "auto" and "text" in target_description.lower()):
        # 使用OCR相关的检测
        bbox = visual_grounding(image_path, target_description)
        if bbox:
            text_content = extract_text(image_path, bbox)
            result["extracted_regions"].append({
                "bbox": bbox,
                "type": "text",
                "content": text_content
            })
    else:
        # 使用对象检测
        detection_result = locate_object(image_path, target_description)
        if detection_result["status"] == "SUCCESS":
            for detection in detection_result["detections"]:
                result["extracted_regions"].append({
                    "bbox": detection["bbox"],
                    "type": "object",
                    "confidence": detection["confidence"],
                    "label": detection["label"]
                })
    
    result["status"] = "SUCCESS" if result["extracted_regions"] else "NOT_FOUND"
    
    return result