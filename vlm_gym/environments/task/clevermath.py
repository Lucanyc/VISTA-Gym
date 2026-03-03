"""
Clever-Math Task implementation for VLM Gym
Handles mathematical reasoning tasks on CLEVR synthetic images
"""

from typing import Tuple, Dict, Any, List, Optional
import re
import json
from pathlib import Path
from collections import Counter

from .vision_qa_task import VisionQATask


class ClevrMathTask(VisionQATask):
    """
    Clever-Math 特定任务
    
    专门处理基于CLEVR合成图像的数学推理任务，包括：
    - 加法运算（添加物体后计数）
    - 减法运算（移除物体后计数）
    - 多步运算（连续的加减操作）
    - 对抗性问题（误导性的数学问题）
    - 物体计数（特定属性的物体数量）
    - 存在性判断（特定物体是否存在）
    
    支持需要视觉理解和数学推理能力的合成场景任务
    """
    
    # 任务类型
    TASK_TYPES = {
        'math_addition': '加法运算',
        'math_subtraction': '减法运算',
        'math_addition_counting': '加法后计数',
        'math_subtraction_counting': '减法后计数',
        'math_subtraction_multihop': '多步减法',
        'math_mixed_operations': '混合运算',
        'math_counting': '物体计数',
        'math_counting_remainder': '剩余计数',
        'math_existence': '存在性判断',
        'math_existence_counting': '存在性计数',
        'math_comparison': '数量比较',
        'math_adversarial_counting': '对抗性计数'
    }
    
    # 模板类型
    TEMPLATE_TYPES = {
        'adversarial': '对抗性问题（包含误导信息）',
        'addition': '加法模板',
        'subtraction': '减法模板',
        'subtraction-multihop': '多步减法模板',
        'counting': '计数模板',
        'comparison': '比较模板',
        'existence': '存在性模板'
    }
    
    # CLEVR物体属性
    CLEVR_COLORS = ['red', 'blue', 'green', 'gray', 'brown', 'purple', 'cyan', 'yellow']
    CLEVR_SHAPES = ['cube', 'sphere', 'cylinder', 'block', 'ball']
    CLEVR_SIZES = ['big', 'small', 'large', 'tiny']
    CLEVR_MATERIALS = ['metallic', 'rubber', 'matte', 'shiny']
    
    def __init__(self, task_id: str, adapter: Any):
        """
        初始化Clever-Math任务
        
        Args:
            task_id: 任务ID
            adapter: Clever-Math数据适配器
        """
        # 调用父类初始化
        super().__init__(task_id, adapter)
        
        # 初始化Clever-Math特定属性
        self.task_type = None
        self.template_type = None
        self.numeric_answer = 0
        self.question_analysis = {}
        self.has_addition = False
        self.has_subtraction = False
        self.has_counting = False
        self.has_existence = False
        self.has_comparison = False
        self.has_remainder = False
        self.is_adversarial = False
        self.is_multihop = False
        self.operation_count = 0
        
        # 获取任务数据
        task_data = adapter.get_task_data(task_id)
        self.dataset_name = task_data.get('dataset', 'clevr_math')
        self.task_type = task_data.get('task', 'math_counting')
    
    @classmethod
    def get_task_id(cls) -> str:
        """获取任务类型ID"""
        return "vlm-gym.clevr-math"
    
    def setup(self) -> Tuple[str, Dict[str, Any]]:
        """设置Clever-Math特定的任务"""
        # 调用父类setup
        task_goal, task_info = super().setup()
        
        # 从metadata中提取Clever-Math特定信息
        self.template_type = self.metadata.get('template', 'unknown')
        self.numeric_answer = self.metadata.get('numeric_answer', 0)
        self.question_analysis = self.metadata.get('question_analysis', {})
        
        # 设置问题特征标志
        self.has_addition = self.question_analysis.get('has_add', False)
        self.has_subtraction = self.question_analysis.get('has_subtract', False)
        self.has_counting = self.question_analysis.get('has_counting', False)
        self.has_existence = self.question_analysis.get('has_existence', False)
        self.has_comparison = self.question_analysis.get('has_comparison', False)
        self.has_remainder = self.question_analysis.get('has_remainder', False)
        
        # 判断特殊任务类型
        self.is_adversarial = self.template_type == 'adversarial'
        self.is_multihop = 'multihop' in self.template_type or 'multihop' in self.task_type
        
        # 计算操作数量
        self.operation_count = sum([
            self.has_addition,
            self.has_subtraction,
            self.has_counting,
            self.has_comparison
        ])
        
        # 分析问题中的物体属性
        object_info = self._analyze_objects_in_question()
        
        # 添加Clever-Math特定的信息
        task_info["task_type"] = self.task_type
        task_info["template_type"] = self.template_type
        task_info["numeric_answer"] = self.numeric_answer
        task_info["is_adversarial"] = self.is_adversarial
        task_info["is_multihop"] = self.is_multihop
        task_info["operation_count"] = self.operation_count
        task_info["question_features"] = {
            "has_addition": self.has_addition,
            "has_subtraction": self.has_subtraction,
            "has_counting": self.has_counting,
            "has_existence": self.has_existence,
            "has_comparison": self.has_comparison,
            "has_remainder": self.has_remainder
        }
        task_info["object_attributes"] = object_info
        task_info["dataset"] = "clevr_math"
        task_info["difficulty"] = self._assess_difficulty()
        
        # 修改任务目标以包含Clever-Math特定指导
        enhanced_goal = self._enhance_task_goal(task_goal, object_info)
        
        return enhanced_goal, task_info
    
    def _enhance_task_goal(self, base_goal: str, object_info: Dict[str, List[str]]) -> str:
        """增强任务目标描述，添加Clever-Math特定的指导"""
        enhanced_parts = [base_goal]
        
        # 添加CLEVR场景理解通用指导
        enhanced_parts.append("\n**CLEVR Scene Understanding:**")
        enhanced_parts.append("1. Identify all objects in the scene")
        enhanced_parts.append("2. Note object properties: color, shape, size, material")
        enhanced_parts.append("3. Count objects with specific attributes carefully")
        enhanced_parts.append("4. Pay attention to spatial relationships")
        
        # 根据操作类型添加特定指导
        if self.has_addition:
            enhanced_parts.append("\n**Addition Operation:**")
            enhanced_parts.append("- First count existing objects matching the criteria")
            enhanced_parts.append("- Add the specified number to your count")
            enhanced_parts.append("- The question asks about the result AFTER addition")
            if self.has_counting:
                enhanced_parts.append("- Count the specific objects mentioned after the operation")
        
        if self.has_subtraction:
            enhanced_parts.append("\n**Subtraction Operation:**")
            enhanced_parts.append("- Identify objects to be subtracted/removed")
            enhanced_parts.append("- Count remaining objects after removal")
            enhanced_parts.append("- Focus on what's LEFT after subtraction")
            if self.is_multihop:
                enhanced_parts.append("- Apply operations in the order specified")
                enhanced_parts.append("- Keep track of intermediate results")
        
        if self.has_counting and not (self.has_addition or self.has_subtraction):
            enhanced_parts.append("\n**Counting Task:**")
            enhanced_parts.append("- Count only objects matching ALL specified criteria")
            enhanced_parts.append("- Be precise about attributes (color, shape, size)")
        
        if self.has_existence:
            enhanced_parts.append("\n**Existence Check:**")
            enhanced_parts.append("- Determine if objects with specified attributes exist")
            enhanced_parts.append("- Count how many exist if asked")
        
        if self.has_comparison:
            enhanced_parts.append("\n**Comparison Task:**")
            enhanced_parts.append("- Count objects in each category")
            enhanced_parts.append("- Compare the quantities as requested")
        
        # 对抗性任务警告
        if self.is_adversarial:
            enhanced_parts.append("\n**⚠️ Adversarial Task Warning:**")
            enhanced_parts.append("- This question may contain misleading information")
            enhanced_parts.append("- Focus on what the question ACTUALLY asks")
            enhanced_parts.append("- Don't be distracted by irrelevant operations")
            enhanced_parts.append("- The answer might be 0 if conditions aren't met")
        
        # 物体属性提示
        if object_info:
            enhanced_parts.append("\n**Objects to Look For:**")
            if object_info.get('colors'):
                enhanced_parts.append(f"- Colors: {', '.join(object_info['colors'])}")
            if object_info.get('shapes'):
                enhanced_parts.append(f"- Shapes: {', '.join(object_info['shapes'])}")
            if object_info.get('sizes'):
                enhanced_parts.append(f"- Sizes: {', '.join(object_info['sizes'])}")
            if object_info.get('materials'):
                enhanced_parts.append(f"- Materials: {', '.join(object_info['materials'])}")
        
        # 答案格式指导
        enhanced_parts.append("\n**Answer Format:**")
        enhanced_parts.append("- Provide a single integer number")
        enhanced_parts.append("- Count can be 0 if no objects match")
        enhanced_parts.append("- No units or additional text needed")
        
        # 常见错误提醒
        if self.operation_count > 1:
            enhanced_parts.append("\n**Common Mistakes to Avoid:**")
            enhanced_parts.append("- Don't skip any operation steps")
            enhanced_parts.append("- Apply operations in the correct order")
            enhanced_parts.append("- Remember intermediate results")
        
        if self.has_remainder:
            enhanced_parts.append("- Focus on what remains/is left after operations")
        
        return "\n".join(enhanced_parts)
    
    def _analyze_objects_in_question(self) -> Dict[str, List[str]]:
        """分析问题中提到的物体属性"""
        if not self.question:
            return {}
        
        q_lower = self.question.lower()
        object_info = {
            'colors': [],
            'shapes': [],
            'sizes': [],
            'materials': []
        }
        
        # 提取颜色
        for color in self.CLEVR_COLORS:
            if color in q_lower:
                object_info['colors'].append(color)
        
        # 提取形状
        for shape in self.CLEVR_SHAPES:
            if shape in q_lower:
                object_info['shapes'].append(shape)
        
        # 提取大小
        for size in self.CLEVR_SIZES:
            if size in q_lower:
                object_info['sizes'].append(size)
        
        # 提取材质
        for material in self.CLEVR_MATERIALS:
            if material in q_lower:
                object_info['materials'].append(material)
        
        # 移除空列表
        object_info = {k: v for k, v in object_info.items() if v}
        
        return object_info
    
    def check_success(self, action: Any) -> Tuple[bool, str]:
        """
        检查Clever-Math答案
        
        只需要检查数值是否正确
        """
        if action is None:
            return False, "No answer provided"
        
        # 提取用户提供的数字
        user_number = self._extract_number_from_answer(action)
        
        if user_number is None:
            return False, "Please provide a numeric answer"
        
        # 获取正确答案
        correct_answer = self.numeric_answer
        
        # 检查是否匹配
        if user_number == correct_answer:
            return True, f"Correct! The answer is {correct_answer}"
        else:
            # 提供有用的反馈
            if self.is_adversarial and user_number != 0 and correct_answer == 0:
                return False, f"Incorrect. This was an adversarial question. The answer is {correct_answer}"
            elif abs(user_number - correct_answer) == 1:
                return False, f"Close, but incorrect. The answer is {correct_answer} (you said {user_number})"
            else:
                return False, f"Incorrect. The answer is {correct_answer} (you said {user_number})"
    
    def _extract_number_from_answer(self, answer: Any) -> Optional[int]:
        """从答案中提取整数"""
        if answer is None:
            return None
        
        answer_str = str(answer).strip()
        
        # 首先尝试直接转换
        try:
            # 处理浮点数，转换为整数
            num = float(answer_str)
            return int(round(num))
        except:
            pass
        
        # 使用正则表达式查找数字
        # 优先查找独立的数字（前后有边界）
        patterns = [
            r'(?:^|[^\d])([-]?\d+)(?:[^\d]|$)',  # 独立的整数
            r'([-]?\d+)',  # 任何整数
        ]
        
        for pattern in patterns:
            match = re.search(pattern, answer_str)
            if match:
                try:
                    return int(match.group(1))
                except:
                    continue
        
        return None
    
    def validate(
        self,
        chat_history: List[Dict],
        observation: Any,
        full_history: Optional[List[Any]] = None
    ) -> Tuple[float, bool, str, Dict[str, Any]]:
        """
        验证Clever-Math任务执行情况
        """
        # 调用父类验证
        reward, done, message, info = super().validate(
            chat_history, observation, full_history
        )
        
        # 添加Clever-Math特定的信息
        info["task_type"] = self.task_type
        info["template_type"] = self.template_type
        info["numeric_answer"] = self.numeric_answer
        info["is_adversarial"] = self.is_adversarial
        info["is_multihop"] = self.is_multihop
        info["operation_count"] = self.operation_count
        info["difficulty"] = self._assess_difficulty()
        
        # 分析错误类型（如果答案错误）
        if not info.get("success", False) and info.get("answer_provided"):
            error_analysis = self._analyze_error(info["answer_provided"])
            info["error_analysis"] = error_analysis
        
        return reward, done, message, info
    
    def _analyze_error(self, user_answer: Any) -> Dict[str, Any]:
        """分析错误类型"""
        error_info = {
            "error_type": "unknown",
            "user_number": None,
            "correct_number": self.numeric_answer,
            "difference": None
        }
        
        user_number = self._extract_number_from_answer(user_answer)
        if user_number is not None:
            error_info["user_number"] = user_number
            error_info["difference"] = user_number - self.numeric_answer
            
            # 分类错误类型
            if self.is_adversarial and user_number > self.numeric_answer:
                error_info["error_type"] = "adversarial_trap"
            elif abs(error_info["difference"]) == 1:
                error_info["error_type"] = "off_by_one"
            elif self.has_subtraction and user_number > self.numeric_answer:
                error_info["error_type"] = "subtraction_error"
            elif self.has_addition and user_number < self.numeric_answer:
                error_info["error_type"] = "addition_error"
            elif user_number == 0 and self.numeric_answer != 0:
                error_info["error_type"] = "existence_error"
            else:
                error_info["error_type"] = "calculation_error"
        else:
            error_info["error_type"] = "non_numeric_answer"
        
        return error_info
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取Clever-Math特定的指标"""
        metrics = super().get_metrics()
        
        metrics.update({
            "task_type": self.task_type,
            "template_type": self.template_type,
            "numeric_answer": self.numeric_answer,
            "is_adversarial": self.is_adversarial,
            "is_multihop": self.is_multihop,
            "operation_count": self.operation_count,
            "difficulty": self._assess_difficulty(),
            "question_features": {
                "has_addition": self.has_addition,
                "has_subtraction": self.has_subtraction,
                "has_counting": self.has_counting,
                "has_existence": self.has_existence,
                "has_comparison": self.has_comparison,
                "has_remainder": self.has_remainder
            }
        })
        
        return metrics
    
    def _assess_difficulty(self) -> str:
        """评估任务难度"""
        difficulty_score = 0
        
        # 基于操作数量
        difficulty_score += self.operation_count
        
        # 对抗性任务更难
        if self.is_adversarial:
            difficulty_score += 2
        
        # 多步任务更难
        if self.is_multihop:
            difficulty_score += 1
        
        # 基于答案值（较大的数字可能意味着更复杂的场景）
        if abs(self.numeric_answer) > 10:
            difficulty_score += 1
        
        # 返回难度等级
        if difficulty_score >= 4:
            return "hard"
        elif difficulty_score >= 2:
            return "medium"
        else:
            return "easy"
    
    def get_observation(self) -> Dict[str, Any]:
        """获取任务观察"""
        obs = super().get_observation()
        
        # 添加Clever-Math特定信息
        obs["scene_type"] = "CLEVR synthetic scene"
        obs["expected_objects"] = "geometric shapes with various colors, sizes, and materials"
        obs["task_focus"] = "mathematical reasoning about object counts"
        
        return obs
    
    def get_task_description(self) -> str:
        """获取任务描述"""
        desc_parts = [
            f"This is a mathematical reasoning task on a CLEVR synthetic image.",
            f"Task type: {self.TASK_TYPES.get(self.task_type, self.task_type)}",
            f"Template: {self.TEMPLATE_TYPES.get(self.template_type, self.template_type)}"
        ]
        
        if self.is_adversarial:
            desc_parts.append("⚠️ This is an adversarial question - read carefully!")
        
        if self.is_multihop:
            desc_parts.append("This requires multiple steps of reasoning.")
        
        desc_parts.append(f"The expected answer is a number: {self.numeric_answer}")
        
        return "\n".join(desc_parts)