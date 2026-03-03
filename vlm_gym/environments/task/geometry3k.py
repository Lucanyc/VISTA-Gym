#task wrappers for geometry3k

from typing import Tuple, Dict, Any, List, Optional
import re
import math

from .vision_qa_task import VisionQATask


class Geometry3KTask(VisionQATask):
    """
    Geometry3K 特定任务
    
    专门处理几何图形相关的视觉问答任务，包括：
    - 角度计算
    - 长度测量
    - 面积计算
    - 几何定理应用
    - 图形识别与分析
    """
    
    @classmethod
    def get_task_id(cls) -> str:
        """获取任务类型ID"""
        return "vlm-gym.geometry3k"
    
    def setup(self) -> Tuple[str, Dict[str, Any]]:
        """设置Geometry3K特定的任务"""
        # 调用父类setup
        task_goal, task_info = super().setup()
        
        # 添加Geometry3K特定的处理
        task_info["shape_types"] = self._detect_shape_types()
        task_info["problem_goal"] = self._detect_problem_goal()
        task_info["requires_calculation"] = self._requires_calculation()
        task_info["theorem_needed"] = self._detect_theorem_needed()
        task_info["dataset"] = "geometry3k"
        
        # 从metadata中提取信息（如果有）
        if hasattr(self, 'metadata') and self.metadata:
            task_info["has_logic_form"] = self.metadata.get("has_logic_form", False)
            task_info["problem_type_graph"] = self.metadata.get("problem_type_graph", [])
            task_info["problem_type_goal"] = self.metadata.get("problem_type_goal", [])
        
        # 修改任务目标以包含几何特定指导
        enhanced_goal = task_goal
        
        if task_info["requires_calculation"]:
            enhanced_goal += "\n\nNote: This question requires geometric calculations."
        
        if task_info["theorem_needed"]:
            enhanced_goal += f"\nConsider using {task_info['theorem_needed']} to solve this problem."
        
        # 添加Geometry3K特定的提示
        enhanced_goal += "\n\nWhen analyzing the geometric figure, please:"
        enhanced_goal += "\n- Identify all shapes and their properties"
        enhanced_goal += "\n- Note any given measurements or angles"
        enhanced_goal += "\n- Apply relevant geometric theorems or formulas"
        enhanced_goal += "\n- Show your reasoning step by step"
        
        # 如果有选择题，添加提示
        if hasattr(self, 'choices') and self.choices:
            enhanced_goal += f"\n\nChoose from: {', '.join(self.choices)}"
        
        return enhanced_goal, task_info
    
    def _detect_shape_types(self) -> List[str]:
        """检测涉及的几何图形类型"""
        if not self.question:
            return []
        
        question_lower = self.question.lower()
        shapes = []
        
        shape_keywords = {
            "triangle": ["triangle", "△", "triangular"],
            "circle": ["circle", "circular", "radius", "diameter"],
            "square": ["square"],
            "rectangle": ["rectangle", "rectangular"],
            "parallelogram": ["parallelogram"],
            "trapezoid": ["trapezoid", "trapezium"],
            "pentagon": ["pentagon", "pentagonal"],
            "hexagon": ["hexagon", "hexagonal"],
            "polygon": ["polygon"],
            "angle": ["angle", "∠"],
            "line": ["line", "segment"],
            "ray": ["ray"]
        }
        
        for shape, keywords in shape_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                shapes.append(shape)
        
        return shapes
    
    def _detect_problem_goal(self) -> str:
        """检测问题目标类型"""
        if not self.question:
            return "unknown"
        
        question_lower = self.question.lower()
        
        # 角度相关
        if any(word in question_lower for word in ["angle", "∠", "degrees", "°"]):
            return "angle"
        
        # 长度相关
        if any(word in question_lower for word in ["length", "distance", "perimeter", "side"]):
            return "length"
        
        # 面积相关
        if any(word in question_lower for word in ["area", "square units"]):
            return "area"
        
        # 体积相关
        if any(word in question_lower for word in ["volume", "cubic"]):
            return "volume"
        
        # 坐标相关
        if any(word in question_lower for word in ["coordinate", "position", "location"]):
            return "coordinate"
        
        # 证明相关
        if any(word in question_lower for word in ["prove", "show that", "verify"]):
            return "proof"
        
        # 查找值
        if any(word in question_lower for word in ["find", "calculate", "determine"]):
            return "find_value"
        
        return "other"
    
    def _requires_calculation(self) -> bool:
        """判断是否需要计算"""
        if not self.question:
            return False
        
        calculation_keywords = [
            "calculate", "find", "determine", "compute",
            "what is", "solve for", "measure of",
            "value of", "equals", "="
        ]
        
        question_lower = self.question.lower()
        return any(keyword in question_lower for keyword in calculation_keywords)
    
    def _detect_theorem_needed(self) -> Optional[str]:
        """检测可能需要的几何定理"""
        if not self.question:
            return None
        
        question_lower = self.question.lower()
        
        # 全等三角形
        if "congruent" in question_lower or "≅" in question_lower:
            return "Congruent Triangles Theorem"
        
        # 相似三角形
        if "similar" in question_lower or "~" in question_lower:
            return "Similar Triangles Theorem"
        
        # 平行线
        if "parallel" in question_lower or "∥" in question_lower:
            return "Parallel Lines Theorem"
        
        # 垂直
        if "perpendicular" in question_lower or "⊥" in question_lower:
            return "Perpendicular Lines Properties"
        
        # 勾股定理
        if "right triangle" in question_lower or "pythagorean" in question_lower:
            return "Pythagorean Theorem"
        
        # 圆相关
        if "circle" in question_lower:
            if "tangent" in question_lower:
                return "Tangent Properties"
            elif "chord" in question_lower:
                return "Chord Properties"
            elif "inscribed" in question_lower:
                return "Inscribed Angle Theorem"
        
        return None
    
    def check_success(self, action: Any) -> Tuple[bool, str]:
        """
        检查Geometry3K答案
        
        对于数值答案，允许一定的误差范围
        """
        if not action:
            return False, "No answer provided"
        
        # 清理答案格式
        action = str(action).strip()
        
        # 首先尝试父类的检查
        success, feedback = super().check_success(action)
        
        # 如果父类检查失败，尝试Geometry3K特定的检查
        if not success and self.answer:
            # 检查数值答案（允许误差）
            user_value = self._extract_number(action)
            correct_value = self._extract_number(str(self.answer))
            
            if user_value is not None and correct_value is not None:
                # 计算绝对误差和相对误差
                abs_error = abs(user_value - correct_value)
                
                # 对于角度，允许0.5度的误差
                if self._detect_problem_goal() == "angle" and abs_error <= 0.5:
                    return True, f"Correct! (angle within 0.5° tolerance)"
                
                # 对于其他数值，允许0.1%的相对误差
                if correct_value != 0:
                    rel_error = abs_error / abs(correct_value)
                    if rel_error <= 0.001:  # 0.1%误差
                        return True, f"Correct! (within acceptable tolerance)"
                elif abs_error <= 0.01:  # 绝对误差检查
                    return True, f"Correct! (exact match)"
                
                return False, f"Incorrect. Expected {correct_value}, got {user_value}"
            
            # 检查单位问题
            if self._check_unit_compatibility(action, str(self.answer)):
                return True, "Correct! (with appropriate units)"
        
        return success, feedback
    
    def _extract_number(self, text: str) -> Optional[float]:
        """从文本中提取数字（支持几何常用格式）"""
        if not text:
            return None
        
        # 清理文本
        text = text.lower().strip()
        
        # 移除单位
        units = ["degrees", "degree", "°", "cm", "m", "mm", "km", 
                "square", "sq", "cubic", "cu", "units"]
        for unit in units:
            text = text.replace(unit, "")
        
        # 处理分数（如 1/2, 3/4）
        fraction_match = re.search(r'(\d+)\s*/\s*(\d+)', text)
        if fraction_match:
            numerator = float(fraction_match.group(1))
            denominator = float(fraction_match.group(2))
            if denominator != 0:
                return numerator / denominator
        
        # 处理π相关的表达式
        if "π" in text or "pi" in text:
            # 例如 "2π" 或 "π/2"
            text = text.replace("π", str(math.pi)).replace("pi", str(math.pi))
        
        # 处理平方根
        sqrt_match = re.search(r'√(\d+)', text)
        if sqrt_match:
            return math.sqrt(float(sqrt_match.group(1)))
        
        # 查找普通数字
        number_pattern = r'-?\d+\.?\d*'
        matches = re.findall(number_pattern, text)
        
        if matches:
            try:
                return float(matches[0])
            except ValueError:
                pass
        
        return None
    
    def _check_unit_compatibility(self, user_answer: str, correct_answer: str) -> bool:
        """检查单位兼容性"""
        user_value = self._extract_number(user_answer)
        correct_value = self._extract_number(correct_answer)
        
        if user_value is None or correct_value is None:
            return False
        
        # 角度单位转换（度和弧度）
        if abs(user_value * 180/math.pi - correct_value) < 0.01:  # 弧度转度
            return True
        if abs(user_value * math.pi/180 - correct_value) < 0.01:  # 度转弧度
            return True
        
        # 长度单位转换
        length_conversions = {
            10: "mm to cm",
            100: "cm to m", 
            1000: "m to km"
        }
        
        for factor, desc in length_conversions.items():
            if abs(user_value * factor - correct_value) < 0.01 * correct_value:
                return True
            if abs(user_value / factor - correct_value) < 0.01 * correct_value:
                return True
        
        return False
    
    def validate(
        self,
        chat_history: List[Dict],
        observation: Any,
        full_history: Optional[List[Any]] = None
    ) -> Tuple[float, bool, str, Dict[str, Any]]:
        """
        验证Geometry3K任务执行情况
        """
        # 调用父类验证
        reward, done, message, info = super().validate(
            chat_history, observation, full_history
        )
        
        # 添加Geometry3K特定的信息
        info["shape_types"] = self._detect_shape_types()
        info["problem_goal"] = self._detect_problem_goal()
        info["required_calculation"] = self._requires_calculation()
        info["theorem_needed"] = self._detect_theorem_needed()
        
        # 如果是数值答案，添加提取的数值信息
        if info.get("answer_provided"):
            extracted_value = self._extract_number(str(info["answer_provided"]))
            if extracted_value is not None:
                info["extracted_numerical_value"] = extracted_value
        
        return reward, done, message, info
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取Geometry3K特定的指标"""
        metrics = super().get_metrics()
        
        metrics.update({
            "shape_types": self._detect_shape_types(),
            "problem_goal": self._detect_problem_goal(),
            "requires_calculation": self._requires_calculation(),
            "theorem_needed": self._detect_theorem_needed(),
            "is_numerical_answer": self._extract_number(str(self.answer)) is not None if self.answer else False,
            "problem_complexity": self._assess_complexity(),
            "has_multiple_choice": bool(hasattr(self, 'choices') and self.choices)
        })
        
        return metrics
    
    def _assess_complexity(self) -> str:
        """评估问题复杂度"""
        if not self.question:
            return "unknown"
        
        question_lower = self.question.lower()
        complexity_score = 0
        
        # 复杂特征
        if self._detect_theorem_needed():
            complexity_score += 2
        
        # 多步骤标志
        multi_step_keywords = ["then", "after", "next", "finally", "given that"]
        complexity_score += sum(1 for k in multi_step_keywords if k in question_lower)
        
        # 多个形状
        shape_count = len(self._detect_shape_types())
        if shape_count > 2:
            complexity_score += 2
        elif shape_count > 1:
            complexity_score += 1
        
        # 证明问题通常更复杂
        if self._detect_problem_goal() == "proof":
            complexity_score += 3
        
        # 分类
        if complexity_score >= 4:
            return "high"
        elif complexity_score >= 2:
            return "medium"
        else:
            return "low"