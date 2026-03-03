# vlm_gym/environments/tools/geometry_solver.py

import json
import re
import requests
from typing import Optional, Dict, Any
from PIL import Image
from .base import ToolBase

class GeometrySolverTool(ToolBase):
    """几何问题组合求解器 - 内部顺序调用DF→MM"""
    
    # 类级别属性（必需）
    name = "geometry_solver"
    description = "几何问题综合求解器（DiagramFormalizer + MultiMath）"
    capabilities = ["几何问题求解", "CDL提取", "数学推理"]
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化工具"""
        super().__init__(config)
        
        self.config = config or {}
        self.df_tool = self.config.get('df_tool')
        if self.df_tool is None:
            raise ValueError("DiagramFormalizer tool is required for GeometrySolverTool")
        
        self.mm_config = self.config.get('mm_config', {
            "api_url": "http://localhost:8001",
            "timeout": 60
        })
        
        self.debug = self.config.get('debug', False)
        self.current_image = None
        self.current_image_path = None
    
    def reset(self, image=None):
        """重置工具状态"""
        if isinstance(image, str):
            self.current_image_path = image
            self.current_image = Image.open(image)
        elif isinstance(image, Image.Image):
            self.current_image = image
            self.current_image_path = getattr(image, 'filename', None)
        else:
            self.current_image = None
            self.current_image_path = None
        
        # 同时重置DF工具
        if self.df_tool and hasattr(self.df_tool, 'reset') and self.current_image:
            self.df_tool.reset(self.current_image)
            if self.debug:
                print(f"  ✓ Reset DF tool with image")
    
    def execute(self, action_string) -> Dict[str, Any]:
        """执行工具"""
        if isinstance(action_string, str):
            try:
                params = json.loads(action_string)
            except json.JSONDecodeError:
                params = {"question": action_string}
        else:
            params = action_string
        
        image_path = params.get('image_path', self.current_image_path)
        question = params.get('question', params.get('problem', ''))
        
        if not question:
            return {
                "error": "Question is required",
                "success": False
            }
        
        return self.invoke(image_path, question)
    
    def invoke(self, image_path: str, question: str) -> Dict[str, Any]:
        """执行组合工具：DF → MM → Answer"""
        execution_trace = []
        
        try:
            # Step 1: 调用DiagramFormalizer
            if self.debug:
                print(f"\n[GeometrySolver] Step 1: Calling DiagramFormalizer")
                print(f"  Image: {image_path}")
                print(f"  Question: {question[:100]}...")
            
            # ⭐ 关键修改：确保DF工具已经通过reset设置了图像
            if image_path and (not self.current_image or self.current_image_path != image_path):
                # 如果图像路径变了，重新加载并reset
                if self.debug:
                    print(f"  Loading and resetting DF with new image...")
                    
                image = Image.open(image_path)
                self.current_image = image
                self.current_image_path = image_path
                
                # 重置DF工具，这是关键！
                if hasattr(self.df_tool, 'reset'):
                    self.df_tool.reset(image)
                    if self.debug:
                        print(f"  ✓ DF tool reset with image from {image_path}")
            elif self.current_image and hasattr(self.df_tool, 'reset'):
                # 确保DF已经设置了当前图像
                self.df_tool.reset(self.current_image)
                if self.debug:
                    print(f"  ✓ DF tool has current image")
            
            # ⭐ 构建DF输入 - 只需要task和problem，不需要image参数
            # 参考agent代码中的格式
            df_prompt = """Look at the geometric figure in the image. 
Please describe the construction and measurements by predicting the construction_cdl and image_cdl."""
            
            # 可以将用户问题集成到prompt中
            if question and question != "Extract CDL from the geometric figure":
                df_prompt += f"\n\nAdditional context: {question}"
            
            df_input = {
                "task": "extract_cdl",
                "problem": df_prompt
            }
            
            if self.debug:
                print(f"  DF input: {json.dumps(df_input, indent=2)}")
            
            # 执行DF
            df_result = self.df_tool.execute(df_input)
            
            if self.debug:
                print(f"  DF result keys: {list(df_result.keys()) if isinstance(df_result, dict) else 'Not a dict'}")
                print(f"  DF success: {df_result.get('success', False) if isinstance(df_result, dict) else 'N/A'}")
            
            # 检查结果
            if not df_result:
                raise RuntimeError("DiagramFormalizer returned None")
                
            if not df_result.get("success", False):
                error_msg = df_result.get('error', 'Unknown error')
                raise RuntimeError(f"DiagramFormalizer failed: {error_msg}")
            
            # 解析CDL输出
            raw_output = (df_result.get("formalized_output", "") or 
                         df_result.get("raw_response", "") or 
                         df_result.get("output", ""))
                         
            if not raw_output:
                # 尝试从其他字段获取
                if "construction_cdl" in df_result or "image_cdl" in df_result:
                    raw_output = f"construction_cdl: {df_result.get('construction_cdl', '')}\nimage_cdl: {df_result.get('image_cdl', '')}"
                else:
                    raise RuntimeError("DiagramFormalizer returned empty output")
            
            cdl_output = self._parse_cdl(raw_output)
            
            if self.debug:
                print(f"  Raw output preview: {raw_output[:200]}...")
                print(f"  Parsed CDL: {cdl_output}")
            
            execution_trace.append({
                "tool": "diagram_formalizer",
                "input": df_input,
                "output": cdl_output,
                "raw_output": raw_output
            })
            
            if self.debug:
                print(f"  ✓ CDL extracted successfully")
                print(f"  Construction CDL: {cdl_output.get('construction_cdl', 'N/A')[:100]}...")
                print(f"  Image CDL: {cdl_output.get('image_cdl', 'N/A')[:100]}...")
            
            # Step 2: 调用MultiMath Server
            if self.debug:
                print(f"\n[GeometrySolver] Step 2: Calling MultiMath Server")
            
            # 构建更完整的问题描述，包含CDL信息
            enhanced_question = question
            if cdl_output.get('construction_cdl') or cdl_output.get('image_cdl'):
                enhanced_question = f"{question}\n\nGeometric information:\n"
                if cdl_output.get('construction_cdl'):
                    enhanced_question += f"Construction: {cdl_output['construction_cdl']}\n"
                if cdl_output.get('image_cdl'):
                    enhanced_question += f"Measurements: {cdl_output['image_cdl']}"
            
            mm_input = {
                "task": "solve",
                "question": enhanced_question,
                "problem_type": "geometry",
                "output_format": "answer_only",
                "parsed_cdl": cdl_output,
                "image_ref": image_path
            }
            
            mm_output = self._call_multimath(mm_input)
            
            if not mm_output.get("success", False):
                raise RuntimeError(f"MultiMath failed: {mm_output.get('error', 'Unknown error')}")
            
            execution_trace.append({
                "tool": "multimath_server",
                "input": mm_input,
                "output": mm_output
            })
            
            if self.debug:
                print(f"  ✓ MultiMath solved: {mm_output.get('answer', 'N/A')}")
            
            # Step 3: 提取最终答案
            final_answer = self._extract_final_answer(mm_output)
            
            if self.debug:
                print(f"\n[GeometrySolver] Final answer: {final_answer}")
            
            return {
                "success": True,
                "final_answer": final_answer,
                "cdl": cdl_output,
                "multimath_input": mm_input,
                "multimath_output": mm_output,
                "execution_trace": execution_trace,
                "method": "DF+MM Pipeline"
            }
            
        except Exception as e:
            if self.debug:
                print(f"\n[GeometrySolver] Error: {e}")
                import traceback
                traceback.print_exc()
            
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "final_answer": None,
                "cdl": {},
                "execution_trace": execution_trace
            }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """返回工具能力描述"""
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "pipeline": "DiagramFormalizer → MultiMath Server",
            "parameters": {
                "image_path": {
                    "type": "string",
                    "required": False,
                    "description": "几何图形图像路径"
                },
                "question": {
                    "type": "string", 
                    "required": True,
                    "description": "几何问题描述"
                }
            },
            "output": {
                "final_answer": {
                    "type": "string",
                    "description": "最终答案"
                },
                "cdl": {
                    "type": "dict",
                    "description": "提取的CDL形式化表示"
                },
                "execution_trace": {
                    "type": "list",
                    "description": "执行轨迹"
                },
                "success": {
                    "type": "bool",
                    "description": "是否成功"
                }
            }
        }
    
    def _parse_cdl(self, output: str) -> Dict[str, str]:
        """解析CDL输出 - 改进版本"""
        if not output:
            return {"construction_cdl": "", "image_cdl": ""}
        
        cdl_result = {"construction_cdl": "", "image_cdl": ""}
        
        # 更灵活的正则表达式模式
        patterns = {
            'construction_cdl': [
                r'construction_cdl[:\s]+(.*?)(?=image_cdl|$)',
                r'Construction CDL[:\s]+(.*?)(?=Image CDL|$)',
                r'construction[:\s]+(.*?)(?=image|measurements|$)'
            ],
            'image_cdl': [
                r'image_cdl[:\s]+(.*?)(?=$)',
                r'Image CDL[:\s]+(.*?)(?=$)',
                r'measurements?[:\s]+(.*?)(?=$)',
                r'image[:\s]+(.*?)(?=$)'
            ]
        }
        
        # 尝试多个模式
        for key, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
                if match:
                    cdl_result[key] = match.group(1).strip()
                    break
        
        # 如果没有找到标准格式，把整个输出作为image_cdl
        if not cdl_result["construction_cdl"] and not cdl_result["image_cdl"]:
            # 可能整个输出就是CDL描述
            cdl_result["image_cdl"] = output.strip()
        
        return cdl_result
    
    def _call_multimath(self, mm_input: Dict[str, Any]) -> Dict[str, Any]:
        """调用远程MultiMath服务"""
        try:
            response = requests.post(
                f"{self.mm_config['api_url']}/solve",
                json=mm_input,
                timeout=self.mm_config.get('timeout', 60)
            )
            
            if response.status_code == 200:
                result = response.json()
                result["success"] = True
                return result
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "answer": None
                }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": str(e),
                "answer": None
            }
    
    def _extract_final_answer(self, mm_output: Dict[str, Any]) -> Optional[str]:
        """从MultiMath输出提取最终答案"""
        if "answer" in mm_output and mm_output["answer"]:
            return str(mm_output["answer"])
        
        if "solution" in mm_output:
            text = str(mm_output["solution"])
            match = re.search(r'(?:answer|result|y)\s*[:=]\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None