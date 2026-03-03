#!/usr/bin/env python3
"""
UniGeo Task implementation for VLM Gym
Handles geometric calculation problems with visual diagrams
"""

from typing import Tuple, Dict, Any, List, Optional
import re
import logging

from .vision_qa_task import VisionQATask

logger = logging.getLogger(__name__)


class UniGeoTask(VisionQATask):
    """
    UniGeo 特定任务
    
    专门处理几何计算问题，支持：
    - Multiple choice questions with 4 options
    - Placeholder replacement (N_0, N_1, etc.)
    - Geometric reasoning and calculation
    - Knowledge point-based categorization
    
    评分机制：
    - Multiple choice: exact match with correct choice (both index and value)
    - Support for both text answers (e.g., "40°") and numeric values (e.g., 40.0)
    """
    
    # UniGeo的知识点分类
    KNOWLEDGE_POINTS = {
        '等腰三角形': 'Isosceles triangle properties',
        '平行线': 'Parallel lines',
        '角平分线': 'Angle bisector',
        '三角形内角和': 'Sum of triangle angles',
        '直角三角形': 'Right triangle',
        '圆周角': 'Inscribed angle',
        '外接圆': 'Circumscribed circle',
        '多边形': 'Polygon',
        '圆内接四边形': 'Cyclic quadrilateral',
        '对称': 'Symmetry',
        '邻补角': 'Supplementary angles',
        '对顶角': 'Vertical angles',
        '立体图形': '3D geometry'
    }
    
    # 常见的几何单位和符号
    GEOMETRY_SYMBOLS = {
        '°': 'degrees',
        '∠': 'angle',
        '△': 'triangle',
        '⊥': 'perpendicular',
        '∥': 'parallel',
        '≅': 'congruent',
        'π': 'pi'
    }
    
    @classmethod
    def get_task_id(cls) -> str:
        """获取任务类型ID"""
        return "vlm-gym.unigeo"
    
    def setup(self) -> Tuple[str, Dict[str, Any]]:
        """设置UniGeo特定的任务"""
        # 调用父类setup
        task_goal, task_info = super().setup()
        
        # 从metadata中提取UniGeo特定信息
        self.original_id = self.metadata.get('original_id', -1)
        self.split = self.metadata.get('split', 'train')
        
        # 答案信息
        self.answer_label = self.metadata.get('answer_label', -1)  # 1-based选项索引
        self.answer_value = self.metadata.get('answer_value', None)  # 数值答案
        self.target_number = self.metadata.get('target_number', None)
        
        # 选择题选项
        self.choices = self.task_data.get('choices', [])
        self.choice_nums = self.metadata.get('choice_nums', [])
        
        # 知识点和解题信息
        self.knowledge_points = self.metadata.get('knowledge_points', [])
        self.manual_program = self.metadata.get('manual_program', [])
        self.solution = self.metadata.get('solution', '')
        
        # 占位符信息
        self.numbers = self.metadata.get('numbers', [])
        self.original_question_with_placeholders = self.metadata.get('original_question_with_placeholders', '')
        
        # 处理占位符（如果question中还有占位符）
        self.processed_question = self._process_placeholders(self.question)
        
        # 分析任务难度
        self.difficulty = self._analyze_difficulty()
        
        # 分析几何类型
        self.geometry_types = self._analyze_geometry_types()
        
        # 添加UniGeo特定信息到task_info
        task_info["original_id"] = self.original_id
        task_info["split"] = self.split
        task_info["difficulty"] = self.difficulty
        task_info["knowledge_points"] = self.knowledge_points
        task_info["geometry_types"] = self.geometry_types
        task_info["has_solution"] = bool(self.solution)
        task_info["program_steps"] = len(self.manual_program)
        task_info["num_placeholders"] = len(self.numbers)
        task_info["dataset"] = "unigeo"
        task_info["task"] = "geometry_calculation"
        
        # 修改任务目标以包含UniGeo特定指导
        enhanced_goal = self._enhance_task_goal(task_goal)
        
        return enhanced_goal, task_info
    
    def _process_placeholders(self, text: str) -> str:
        """处理占位符，将N_0, N_1等替换为实际数值"""
        processed = text
        for i, num in enumerate(self.numbers):
            placeholder = f"N_{i}"
            # 如果是整数，显示为整数格式
            if isinstance(num, float) and num == int(num):
                processed = processed.replace(placeholder, str(int(num)))
            else:
                processed = processed.replace(placeholder, str(num))
        return processed
    
    def _analyze_difficulty(self) -> str:
        """基于程序步骤分析难度"""
        steps = len(self.manual_program)
        if steps == 0:
            return "no_program"
        elif steps <= 2:
            return "easy"
        elif steps <= 5:
            return "medium"
        else:
            return "hard"
    
    def _analyze_geometry_types(self) -> List[str]:
        """分析问题中涉及的几何类型"""
        types = []
        
        # 从知识点提取
        types.extend(self.knowledge_points)
        
        # 从问题文本分析
        question_lower = self.processed_question.lower()
        
        if 'triangle' in question_lower or '△' in self.processed_question:
            types.append('triangle')
        if 'circle' in question_lower or '⊙' in self.processed_question:
            types.append('circle')
        if 'angle' in question_lower or '∠' in self.processed_question:
            types.append('angle')
        if 'parallel' in question_lower or '∥' in self.processed_question:
            types.append('parallel')
        if 'perpendicular' in question_lower or '⊥' in self.processed_question:
            types.append('perpendicular')
        
        return list(set(types))  # 去重
    
    def _enhance_task_goal(self, base_goal: str) -> str:
        """增强任务目标描述，添加UniGeo特定的指导"""
        enhanced_parts = [base_goal]
        
        # 确保使用处理过的问题
        if self.processed_question != self.question:
            enhanced_parts.insert(0, f"**Question:** {self.processed_question}")
        
        # 多选题信息
        enhanced_parts.append("\n**This is a geometry multiple choice question.**")
        enhanced_parts.append("Choose the correct answer from the given options:")
        for i, choice in enumerate(self.choices):
            enhanced_parts.append(f"  {i+1}: {choice}")
        enhanced_parts.append("Provide your answer as either:")
        enhanced_parts.append("- The option number (1-4)")
        enhanced_parts.append("- The exact text of your chosen option (e.g., '40°')")
        
        # 几何问题解题指导
        enhanced_parts.append("\n**Geometry Problem Solving Tips:**")
        
        # 根据知识点提供特定指导
        if self.knowledge_points:
            enhanced_parts.append(f"\nThis problem involves: {', '.join(self.knowledge_points)}")
            
            # 提供知识点相关的提示
            if '等腰三角形' in self.knowledge_points:
                enhanced_parts.append("- Remember: In an isosceles triangle, two sides are equal and their opposite angles are equal")
            if '平行线' in self.knowledge_points:
                enhanced_parts.append("- Remember: Parallel lines create equal corresponding angles and supplementary co-interior angles")
            if '三角形内角和' in self.knowledge_points:
                enhanced_parts.append("- Remember: The sum of angles in a triangle is always 180°")
            if '圆周角' in self.knowledge_points:
                enhanced_parts.append("- Remember: Inscribed angles subtending the same arc are equal")
            if '直角三角形' in self.knowledge_points:
                enhanced_parts.append("- Remember: In a right triangle, one angle is 90° and the other two sum to 90°")
        
        # 根据难度提供建议
        if self.difficulty == "easy":
            enhanced_parts.append("\n**This is a basic problem** - Apply direct geometric properties")
        elif self.difficulty == "medium":
            enhanced_parts.append("\n**This is a moderate problem** - May require 2-3 steps of reasoning")
        elif self.difficulty == "hard":
            enhanced_parts.append("\n**This is a challenging problem** - Requires multiple geometric concepts and careful analysis")
        
        # 通用几何解题建议
        enhanced_parts.append("\n**General approach:**")
        enhanced_parts.append("1. Identify all given information in the figure")
        enhanced_parts.append("2. Determine what geometric properties apply")
        enhanced_parts.append("3. Set up equations based on these properties")
        enhanced_parts.append("4. Solve step by step to find the required value")
        enhanced_parts.append("5. Check if your answer matches one of the options")
        
        return "\n".join(enhanced_parts)
    
    def check_success(self, action: Any) -> Tuple[bool, str]:
        """
        检查UniGeo答案
        
        支持两种答案格式：
        1. 选项索引（1-4）
        2. 选项文本（如 "40°"）
        """
        if not action:
            return False, "No answer provided"
        
        user_answer = str(action).strip()
        
        # 获取正确答案信息
        correct_index = self.answer_label  # 1-based
        correct_text = self.answer if isinstance(self.answer, str) else str(self.answer)
        
        # 检查是否提供了选项索引
        try:
            user_index = int(user_answer)
            if 1 <= user_index <= len(self.choices):
                if user_index == correct_index:
                    return True, f"Correct! The answer is {self.choices[correct_index-1]}"
                else:
                    return False, f"Incorrect. The correct answer is option {correct_index}: {self.choices[correct_index-1]}"
            else:
                return False, f"Invalid option number. Please choose between 1 and {len(self.choices)}"
        except ValueError:
            pass
        
        # 检查是否匹配选项文本
        user_normalized = self._normalize_answer(user_answer)
        
        # 检查每个选项
        for idx, choice in enumerate(self.choices, 1):
            choice_normalized = self._normalize_answer(choice)
            if user_normalized == choice_normalized:
                if idx == correct_index:
                    return True, f"Correct! The answer is {choice}"
                else:
                    return False, f"Incorrect. The correct answer is option {correct_index}: {self.choices[correct_index-1]}"
        
        # 检查是否直接匹配正确答案
        correct_normalized = self._normalize_answer(correct_text)
        if user_normalized == correct_normalized:
            return True, f"Correct! The answer is {correct_text}"
        
        # 如果有数值答案，尝试数值比较
        if self.answer_value is not None:
            try:
                user_value = float(re.sub(r'[^\d.-]', '', user_answer))
                if abs(user_value - self.answer_value) < 0.01:  # 允许小误差
                    return True, f"Correct! The answer is {self.answer_value}"
            except:
                pass
        
        # 提供正确答案
        return False, f"Incorrect. The correct answer is option {correct_index}: {self.choices[correct_index-1]}"
    
    def _normalize_answer(self, answer: str) -> str:
        """标准化答案以便比较"""
        # 转小写
        normalized = answer.lower().strip()
        
        # 统一度数符号
        normalized = normalized.replace('degrees', '°')
        normalized = normalized.replace('degree', '°')
        normalized = normalized.replace(' °', '°')
        
        # 移除多余空格
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # 如果是纯数字加单位，标准化格式
        match = re.match(r'^(\d+\.?\d*)\s*°?$', normalized)
        if match:
            num = float(match.group(1))
            if num == int(num):
                normalized = f"{int(num)}°"
            else:
                normalized = f"{num}°"
        
        return normalized
    
    def validate(
        self,
        chat_history: List[Dict],
        observation: Any,
        full_history: Optional[List[Any]] = None
    ) -> Tuple[float, bool, str, Dict[str, Any]]:
        """
        验证UniGeo任务执行情况
        """
        # 调用父类验证
        reward, done, message, info = super().validate(
            chat_history, observation, full_history
        )
        
        # 添加UniGeo特定的信息
        info["original_id"] = self.original_id
        info["difficulty"] = self.difficulty
        info["knowledge_points"] = self.knowledge_points
        info["geometry_types"] = self.geometry_types
        info["program_steps"] = len(self.manual_program)
        info["has_solution"] = bool(self.solution)
        
        # 如果提供了答案，进行详细分析
        if info.get("answer_provided"):
            provided_answer = str(info["answer_provided"])
            
            # 尝试解析选项索引
            try:
                choice_idx = int(provided_answer)
                if 1 <= choice_idx <= len(self.choices):
                    info["selected_choice_idx"] = choice_idx
                    info["selected_choice_text"] = self.choices[choice_idx-1]
                    info["selected_correct"] = (choice_idx == self.answer_label)
            except:
                # 尝试匹配文本
                normalized_provided = self._normalize_answer(provided_answer)
                for idx, choice in enumerate(self.choices, 1):
                    if self._normalize_answer(choice) == normalized_provided:
                        info["selected_choice_idx"] = idx
                        info["selected_choice_text"] = choice
                        info["selected_correct"] = (idx == self.answer_label)
                        break
            
            # 检查数值精度
            if self.answer_value is not None:
                try:
                    provided_value = float(re.sub(r'[^\d.-]', '', provided_answer))
                    info["provided_numeric_value"] = provided_value
                    info["numeric_error"] = abs(provided_value - self.answer_value)
                except:
                    pass
        
        return reward, done, message, info
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取UniGeo特定的指标"""
        metrics = super().get_metrics()
        
        metrics.update({
            "original_id": self.original_id,
            "split": self.split,
            "difficulty": self.difficulty,
            "knowledge_points": self.knowledge_points,
            "geometry_types": self.geometry_types,
            "num_choices": len(self.choices),
            "program_steps": len(self.manual_program),
            "num_placeholders": len(self.numbers),
            "has_solution": bool(self.solution),
            "answer_type": "multiple_choice",
            "difficulty_factors": self.get_difficulty_factors()
        })
        
        return metrics
    
    def get_difficulty_factors(self) -> List[str]:
        """获取影响难度的因素"""
        factors = []
        
        # 基于步骤数
        if len(self.manual_program) > 5:
            factors.append("multi_step_solution")
        
        # 基于占位符数量
        if len(self.numbers) >= 3:
            factors.append("multiple_values")
        
        # 基于知识点
        if len(self.knowledge_points) > 2:
            factors.append("multiple_concepts")
        
        # 特定的难知识点
        hard_concepts = ['立体图形', '圆内接四边形', '对称']
        if any(kp in hard_concepts for kp in self.knowledge_points):
            factors.append("advanced_concepts")
        
        # 基于问题长度
        if len(self.processed_question.split()) > 25:
            factors.append("complex_description")
        
        return factors
    
    def get_hint(self) -> str:
        """获取问题提示（用于辅助）"""
        hints = []
        
        # 基本提示
        hints.append("Analyze the geometric figure carefully.")
        
        # 根据知识点提供提示
        if '等腰三角形' in self.knowledge_points:
            hints.append("Look for equal sides or angles in the triangle.")
        elif '平行线' in self.knowledge_points:
            hints.append("Use properties of parallel lines and transversals.")
        elif '圆周角' in self.knowledge_points:
            hints.append("Consider inscribed angle theorems.")
        
        # 根据难度提供策略
        if self.difficulty == "hard":
            hints.append("This problem may require combining multiple geometric properties.")
        elif self.difficulty == "medium":
            hints.append("Try to identify 2-3 key relationships in the figure.")
        
        # 如果有多个数值
        if len(self.numbers) > 1:
            hints.append(f"Use the given values: {', '.join(map(str, self.numbers))}")
        
        return " ".join(hints) if hints else "Study the figure and apply relevant geometric principles."
    
    def get_solution_outline(self) -> Optional[str]:
        """获取解题大纲（如果可用）"""
        if not self.manual_program:
            return None
        
        outline_parts = [f"This problem can be solved in {len(self.manual_program)} steps:"]
        
        # 解析程序步骤（简化版）
        for i, step in enumerate(self.manual_program, 1):
            if step.startswith('g_minus'):
                outline_parts.append(f"Step {i}: Calculate difference")
            elif step.startswith('g_half'):
                outline_parts.append(f"Step {i}: Find half/midpoint")
            elif step.startswith('g_double'):
                outline_parts.append(f"Step {i}: Double a value")
            elif step.startswith('g_'):
                outline_parts.append(f"Step {i}: Apply geometric operation")
            else:
                outline_parts.append(f"Step {i}: Use value {step}")
        
        return "\n".join(outline_parts)