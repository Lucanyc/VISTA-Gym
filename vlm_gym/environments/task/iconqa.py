"""
IconQA Task implementation for VLM Gym
Handles visual question answering tasks for pre-K to elementary school levels
"""

from typing import Tuple, Dict, Any, List, Optional
import re

from .vision_qa_task import VisionQATask


class IconQATask(VisionQATask):
    """
    IconQA 特定任务
    
    专门处理面向儿童的视觉问答任务，包括：
    - 计数问题 (C标签)
    - 视觉推理 (V标签)
    - 几何形状 (G标签)
    - 其他数学概念 (E标签等)
    
    支持两种任务类型：
    - choose_txt: 文本选项
    - choose_img: 图像选项（已合并为单张图片）
    """
    
    # IconQA的标签类别及其含义
    LABEL_CATEGORIES = {
        'C': 'counting',         # 计数相关
        'V': 'visual',          # 视觉推理
        'G': 'geometry',        # 几何形状
        'E': 'other',           # 其他数学概念
        'S': 'spatial',         # 空间关系
        'P': 'pattern',         # 模式识别
        'M': 'measurement',     # 测量
        'A': 'arithmetic'       # 算术
    }
    
    # 年级顺序（用于复杂度评估）
    GRADE_ORDER = ['prek', 'kindergarten', 'grade1', 'grade2', 'grade3']
    
    @classmethod
    def get_task_id(cls) -> str:
        """获取任务类型ID"""
        return "vlm-gym.iconqa"
    
    def setup(self) -> Tuple[str, Dict[str, Any]]:
        """设置IconQA特定的任务"""
        # 调用父类setup
        task_goal, task_info = super().setup()
        
        # 从task_data中提取答案信息
        self.answer_letter = self.task_data.get('answer_letter', '') 
        
        # 从metadata中提取IconQA特定信息
        self.task_type = self.metadata.get('task_type', 'choose_txt')
        self.grade = self.metadata.get('grade', 'unknown')
        self.label = self.metadata.get('label', '')
        self.num_choices = self.metadata.get('num_choices', len(self.choices))
        
        # 添加IconQA特定的处理
        task_info["task_type"] = self.task_type
        task_info["grade"] = self.grade
        task_info["label"] = self.label
        task_info["label_category"] = self._get_label_category()
        task_info["question_category"] = self._classify_question_category()
        task_info["difficulty_level"] = self._assess_difficulty()
        task_info["dataset"] = "iconqa"
        
        # 修改任务目标以包含IconQA特定指导
        enhanced_goal = self._enhance_task_goal(task_goal)
        
        return enhanced_goal, task_info
    
    def _enhance_task_goal(self, base_goal: str) -> str:
        """增强任务目标描述，添加IconQA特定的指导"""
        enhanced_parts = [base_goal]
        
        # 根据任务类型添加说明
        if self.task_type == 'choose_img':
            enhanced_parts.append("\nNote: The answer choices are shown as images in the provided picture.")
            enhanced_parts.append("Please select the correct image option (A, B, C, etc.) based on the question.")
        
        # 根据问题类别添加提示
        category = self._classify_question_category()
        
        if category == 'counting':
            enhanced_parts.append("\nThis is a counting question. Count carefully and select the correct number.")
        elif category == 'shape_recognition':
            enhanced_parts.append("\nThis question is about shapes. Identify the shapes and their properties.")
        elif category == 'pattern':
            enhanced_parts.append("\nThis is a pattern recognition question. Look for the pattern or rule.")
        elif category == 'comparison':
            enhanced_parts.append("\nThis question requires comparison. Compare the items carefully.")
        elif category == 'spatial':
            enhanced_parts.append("\nThis question involves spatial relationships. Consider positions and orientations.")
        elif category == 'arithmetic':
            enhanced_parts.append("\nThis is a simple arithmetic question. Perform the calculation carefully.")
        
        # 根据年级添加适当的语言提示
        if self.grade in ['prek', 'kindergarten']:
            enhanced_parts.append("\nRemember: This is designed for young children. The answer should be simple and clear.")
        
        return "\n".join(enhanced_parts)
    
    def _get_label_category(self) -> str:
        """获取标签类别"""
        if not self.label:
            return 'unknown'
        
        # 获取标签的第一个字符
        label_prefix = self.label[0].upper()
        return self.LABEL_CATEGORIES.get(label_prefix, 'other')
    
    def _classify_question_category(self) -> str:
        """分类问题类别"""
        if not self.question:
            return 'unknown'
        
        question_lower = self.question.lower()
        
        # 计数问题
        counting_keywords = ['how many', 'count', 'number of', 'total']
        if any(keyword in question_lower for keyword in counting_keywords):
            return 'counting'
        
        # 形状识别
        shape_keywords = ['shape', 'circle', 'square', 'triangle', 'rectangle', 'hexagon', 'pentagon']
        if any(keyword in question_lower for keyword in shape_keywords):
            return 'shape_recognition'
        
        # 模式识别
        pattern_keywords = ['pattern', 'next', 'sequence', 'continue', 'missing']
        if any(keyword in question_lower for keyword in pattern_keywords):
            return 'pattern'
        
        # 比较问题
        comparison_keywords = ['more', 'less', 'fewer', 'most', 'least', 'bigger', 'smaller', 
                              'longer', 'shorter', 'heavier', 'lighter', 'same', 'different']
        if any(keyword in question_lower for keyword in comparison_keywords):
            return 'comparison'
        
        # 空间关系
        spatial_keywords = ['above', 'below', 'left', 'right', 'between', 'inside', 'outside',
                           'top', 'bottom', 'front', 'back', 'turn', 'flip', 'slide', 'rotate']
        if any(keyword in question_lower for keyword in spatial_keywords):
            return 'spatial'
        
        # 对称性
        if 'symmetry' in question_lower or 'symmetrical' in question_lower:
            return 'symmetry'
        
        # 算术运算
        arithmetic_keywords = ['add', 'subtract', 'plus', 'minus', 'sum', 'difference', '+', '-']
        if any(keyword in question_lower for keyword in arithmetic_keywords):
            return 'arithmetic'
        
        # 颜色识别
        color_keywords = ['color', 'red', 'blue', 'green', 'yellow', 'orange', 'purple', 
                         'pink', 'black', 'white', 'brown']
        if any(keyword in question_lower for keyword in color_keywords):
            return 'color_recognition'
        
        # 分类问题
        if 'which' in question_lower or 'what' in question_lower:
            return 'identification'
        
        return 'other'
    
    def _assess_difficulty(self) -> str:
        """评估问题难度"""
        difficulty_score = 0
        
        # 基于年级
        if self.grade in self.GRADE_ORDER:
            difficulty_score += self.GRADE_ORDER.index(self.grade)
        
        # 基于选项数量
        if self.num_choices > 4:
            difficulty_score += 2
        elif self.num_choices > 3:
            difficulty_score += 1
        
        # 基于问题类别
        category = self._classify_question_category()
        if category in ['pattern', 'spatial', 'arithmetic']:
            difficulty_score += 2
        elif category in ['comparison', 'symmetry']:
            difficulty_score += 1
        
        # 基于问题长度
        if len(self.question) > 50:
            difficulty_score += 1
        
        # 基于任务类型
        if self.task_type == 'choose_img':
            difficulty_score += 1  # 图像选项通常更难
        
        # 分类
        if difficulty_score >= 5:
            return 'hard'
        elif difficulty_score >= 3:
            return 'medium'
        else:
            return 'easy'
    
    
    
    def check_success(self, action: Any) -> Tuple[bool, str]:
        """
        检查IconQA答案 - 支持直接答案和字母答案
        """
        if not action:
            return False, "No answer provided"
        
        # 清理答案格式
        action_str = str(action).strip()
        
        # 获取正确答案的多种形式
        correct_answer_content = self.answer  # 实际答案内容（如 "10"）
        correct_answer_letter = self.task_data.get('answer_letter', '')  # 字母答案（如 "E"）
        
        # 1. 检查直接答案匹配（优先）
        if correct_answer_content and action_str.lower() == correct_answer_content.lower():
            return True, f"Correct! The answer is '{correct_answer_content}'"
        
        # 2. 检查字母答案匹配（单独的字母）
        if correct_answer_letter and len(action_str) == 1 and action_str.upper().isalpha():
            if action_str.upper() == correct_answer_letter:
                return True, f"Correct! {correct_answer_letter} - {correct_answer_content}"
        
        # 3. 检查答案是否包含在句子中
        if correct_answer_content:
            # 检查内容答案
            import re
            if re.search(r'\b' + re.escape(str(correct_answer_content)) + r'\b', action_str, re.IGNORECASE):
                return True, f"Correct! You identified the answer: '{correct_answer_content}'"
        
        # 4. 检查字母答案是否包含在句子中（如 "The answer is E"）
        if correct_answer_letter:
            # 查找句子中的字母答案
            letter_pattern = r'\b' + correct_answer_letter + r'\b'
            if re.search(letter_pattern, action_str.upper()):
                return True, f"Correct! You identified option {correct_answer_letter} - '{correct_answer_content}'"
        
        # 5. 检查是否选择了其他选项（错误情况）
        for i, choice in enumerate(self.choices):
            if str(choice).lower() == action_str.lower():
                choice_letter = chr(65 + i)
                if choice == correct_answer_content:
                    return True, f"Correct! {choice_letter} - {choice}"
                else:
                    return False, f"Incorrect. You selected '{choice}' ({choice_letter}), but the correct answer is '{correct_answer_content}' ({correct_answer_letter})"
        
        # 6. 检查是否是错误的字母
        if len(action_str) == 1 and action_str.upper().isalpha():
            selected_index = ord(action_str.upper()) - ord('A')
            if 0 <= selected_index < len(self.choices):
                selected_choice = self.choices[selected_index]
                return False, f"Incorrect. You selected {action_str.upper()} - '{selected_choice}', but the correct answer is {correct_answer_letter} - '{correct_answer_content}'"
        
        # 默认错误信息
        return False, f"Incorrect. The correct answer is '{correct_answer_content}' (option {correct_answer_letter})"
        
    
    
    def validate(
        self,
        chat_history: List[Dict],
        observation: Any,
        full_history: Optional[List[Any]] = None
    ) -> Tuple[float, bool, str, Dict[str, Any]]:
        """
        验证IconQA任务执行情况
        """
        # 调用父类验证
        reward, done, message, info = super().validate(
            chat_history, observation, full_history
        )
        
        # 添加IconQA特定的信息
        info["task_type"] = self.task_type
        info["grade"] = self.grade
        info["label"] = self.label
        info["label_category"] = self._get_label_category()
        info["question_category"] = self._classify_question_category()
        info["difficulty_level"] = self._assess_difficulty()
        info["num_choices"] = self.num_choices
        
        # 分析答案质量
        if info.get("answer_provided"):
            provided_answer = str(info["answer_provided"]).upper()
            
            # 检查答案格式
            if len(provided_answer) == 1 and provided_answer.isalpha():
                info["answer_format_correct"] = True
                # 检查答案是否在有效范围内
                answer_index = ord(provided_answer) - ord('A')
                if answer_index >= self.num_choices:
                    info["answer_out_of_range"] = True
            else:
                info["answer_format_correct"] = False
                info["format_issue"] = "Expected single letter answer (A, B, C, etc.)"
        
        return reward, done, message, info
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取IconQA特定的指标"""
        metrics = super().get_metrics()
        
        metrics.update({
            "task_type": self.task_type,
            "grade": self.grade,
            "grade_level": self.GRADE_ORDER.index(self.grade) if self.grade in self.GRADE_ORDER else -1,
            "label": self.label,
            "label_category": self._get_label_category(),
            "question_category": self._classify_question_category(),
            "difficulty_level": self._assess_difficulty(),
            "num_choices": self.num_choices,
            "is_visual_choice": self.task_type == 'choose_img',
            "question_complexity": self._assess_question_complexity()
        })
        
        return metrics
    
    def _assess_question_complexity(self) -> str:
        """评估问题复杂度（基于多个因素）"""
        if not self.question:
            return "unknown"
        
        complexity_score = 0
        
        # 问题长度
        if len(self.question) > 50:
            complexity_score += 2
        elif len(self.question) > 30:
            complexity_score += 1
        
        # 是否包含多个概念
        question_lower = self.question.lower()
        concept_keywords = ['and', 'or', 'both', 'all', 'each', 'total', 'together']
        complexity_score += sum(1 for keyword in concept_keywords if keyword in question_lower)
        
        # 是否需要多步推理
        multistep_indicators = ['then', 'after', 'before', 'if', 'when']
        if any(indicator in question_lower for indicator in multistep_indicators):
            complexity_score += 2
        
        # 基于问题类别
        category = self._classify_question_category()
        if category in ['pattern', 'spatial', 'arithmetic']:
            complexity_score += 1
        
        # 分类
        if complexity_score >= 4:
            return "complex"
        elif complexity_score >= 2:
            return "moderate"
        else:
            return "simple"
    
    def get_hint(self) -> str:
        """获取问题提示（用于教育目的）"""
        category = self._classify_question_category()
        
        hints = {
            'counting': "Count each item carefully. Don't miss any!",
            'shape_recognition': "Look at the shapes carefully. What makes them different?",
            'pattern': "Look for what repeats or changes in order.",
            'comparison': "Compare the items one by one.",
            'spatial': "Think about where things are located.",
            'symmetry': "Look for mirror images or equal parts.",
            'arithmetic': "Do the math step by step.",
            'color_recognition': "Look at the colors of each item.",
            'identification': "Look at each option carefully."
        }
        
        return hints.get(category, "Look at the image carefully and think about what the question is asking.")