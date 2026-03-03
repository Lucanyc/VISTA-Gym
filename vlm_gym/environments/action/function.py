
# vlm_gym/environments/action/function.py
"""
通用系统级动作函数
"""
import os
import json
import subprocess
import traceback
from typing import Dict, Any, Optional, List
from pathlib import Path

def request_info(file_path: str, info_type: str, keyterm: str = "") -> str:
    """
    请求文件或目录的特定信息
    
    Examples:
        request_info("/path/to/dataset/", "list_files", "")
        request_info("/path/to/image.jpg", "metadata", "")
        request_info("/path/to/annotations.json", "search", "car")
    """
    results = {}
    
    if info_type == "list_files":
        try:
            if os.path.isdir(file_path):
                files = os.listdir(file_path)
                results = f"Files in directory: {files}"
            else:
                results = "Path is not a directory"
        except Exception as e:
            results = f"Error accessing path: {e}"
            
    elif info_type == "metadata":
        try:
            if file_path.endswith(('.jpg', '.png', '.jpeg')):
                from PIL import Image
                with Image.open(file_path) as img:
                    results = {
                        "size": img.size,
                        "mode": img.mode,
                        "format": img.format
                    }
            else:
                file_stats = os.stat(file_path)
                results = {
                    "size": file_stats.st_size,
                    "modified": file_stats.st_mtime
                }
        except Exception as e:
            results = f"Error getting metadata: {e}"
            
    elif info_type == "search" and keyterm:
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # 简单的关键词搜索
                    matches = []
                    for key, value in data.items():
                        if keyterm.lower() in str(value).lower():
                            matches.append({key: value})
                    results = f"Found {len(matches)} matches for '{keyterm}'"
            else:
                results = "Search only supported for JSON files"
        except Exception as e:
            results = f"Error searching file: {e}"
    
    return json.dumps(results, indent=2) if isinstance(results, dict) else results

def validate_response(response: str, expected_format: str) -> Dict[str, Any]:
    """
    验证响应格式
    
    Examples:
        validate_response("3 cars", "number")
        validate_response("red, blue, green", "list")
        validate_response('{"count": 5}', "json")
    """
    valid_formats = ["number", "list", "text", "json", "boolean"]
    if expected_format not in valid_formats:
        raise ValueError(f"Invalid format. Must be one of {valid_formats}")
    
    result = {
        "response": response,
        "expected_format": expected_format,
        "valid": False,
        "parsed_value": None
    }
    
    if expected_format == "number":
        import re
        numbers = re.findall(r'\d+\.?\d*', response)
        if numbers:
            result["valid"] = True
            result["parsed_value"] = float(numbers[0])
            
    elif expected_format == "list":
        if "," in response:
            result["valid"] = True
            result["parsed_value"] = [item.strip() for item in response.split(",")]
            
    elif expected_format == "json":
        try:
            result["parsed_value"] = json.loads(response)
            result["valid"] = True
        except:
            pass
            
    elif expected_format == "boolean":
        lower_response = response.lower()
        if "yes" in lower_response or "true" in lower_response:
            result["valid"] = True
            result["parsed_value"] = True
        elif "no" in lower_response or "false" in lower_response:
            result["valid"] = True
            result["parsed_value"] = False
    else:
        result["valid"] = True
        result["parsed_value"] = response
    
    return result

def save_result(content: Any, output_path: str, format: str = "json") -> Dict[str, Any]:
    """
    保存结果到文件
    
    Examples:
        save_result({"count": 5}, "/path/to/output.json")
        save_result("Results text", "/path/to/output.txt", "text")
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(content, f, indent=2)
        else:
            with open(output_path, 'w') as f:
                f.write(str(content))
                
        return {
            "status": "SUCCESS",
            "path": output_path,
            "format": format
        }
    except Exception as e:
        return {
            "status": "FAILED",
            "error": str(e)
        }

def debug_info(action_name: str, params: Dict, error: str) -> str:
    """
    生成调试信息
    
    Examples:
        debug_info("analyze_image", {"image_path": "/path/img.jpg"}, "File not found")
    """
    debug_msg = f"""
Debug Information:
- Action: {action_name}
- Parameters: {json.dumps(params, indent=2)}
- Error: {error}

Suggestions:
1. Check if the file path exists
2. Verify parameter formats
3. Ensure required dependencies are installed
"""
    return debug_msg