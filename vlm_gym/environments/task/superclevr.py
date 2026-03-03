"""
SuperCLEVR Task implementation for VLM Gym
Handles visual reasoning tasks on CLEVR synthetic images with yes/no questions
"""

from typing import Tuple, Dict, Any, List, Optional
import re
import json
from pathlib import Path
from collections import Counter

from .vision_qa_task import VisionQATask


class SuperClevrTask(VisionQATask):
    """
    SuperCLEVR 特定任务
    
    专门处理基于CLEVR合成图像的视觉推理任务，包括：
    - 属性比较（大小、颜色、形状、材质的相同性判断）
    - 数量比较（物体数量是否相等）
    - 物体计数（特定属性的物体数量）
    - 存在性判断（特定物体是否存在）
    - 空间推理（左右、前后等空间关系）
    
    支持需要视觉理解和逻辑推理能力的二元判断任务
    """
    
    # 任务类型
    TASK_TYPES = {
        'visual_comparison': '视觉属性比较',
        'counting_comparison': '数量比较',
        'object_counting': '物体计数',
        'object_existence': '存在性判断',
        'spatial_reasoning': '空间关系推理',
        'visual_reasoning': '通用视觉推理'
    }
    
    # 问题类型
    QUESTION_TYPES = {
        'attribute_comparison': '属性比较（颜色、大小、形状、材质）',
        'quantity_comparison': '数量比较（是否相等）',
        'counting': '计数问题',
        'existence': '存在性问题',
        'spatial_relation': '空间关系问题',
        'general': '通用问题'
    }
    
    # CLEVR物体属性
    CLEVR_COLORS = ['red', 'blue', 'green', 'gray', 'brown', 'purple', 'cyan', 'yellow']
    CLEVR_SHAPES = ['cube', 'sphere', 'cylinder', 'ball', 'block']
    CLEVR_SIZES = ['big', 'small', 'large', 'tiny']
    CLEVR_MATERIALS = ['metallic', 'rubber', 'matte', 'shiny']
    CLEVR_OBJECTS = ['chopper', 'scooter', 'bus', 'bike', 'road bike', 'thing', 'object']
    CLEVR_SPATIAL = ['left', 'right', 'behind', 'front', 'above', 'below']
    
    def __init__(self, task_id: str, adapter: Any):
        """
        初始化SuperCLEVR任务
        
        Args:
            task_id: 任务ID
            adapter: SuperCLEVR数据适配器
        """
        # 调用父类初始化
        super().__init__(task_id, adapter)
        
        # 初始化SuperCLEVR特定属性
        self.task_type = None
        self.question_type = None
        self.expected_answer = None
        self.is_binary = True
        self.question_analysis = {}
        self.attributes = {}
        self.complexity_features = {}
        self.is_complex = False
        
        # 获取任务数据
        task_data = adapter.get_task_data(task_id)
        self.dataset_name = task_data.get('dataset', 'superclevr')
        self.task_type = task_data.get('task', 'visual_reasoning')
        self.choices = task_data.get('choices', ['yes', 'no'])
    
    @classmethod
    def get_task_id(cls) -> str:
        """获取任务类型ID"""
        return "vlm-gym.superclevr"
    
    def setup(self) -> Tuple[str, Dict[str, Any]]:
        """设置SuperCLEVR特定的任务"""
        # 调用父类setup
        task_goal, task_info = super().setup()
        
        # 从metadata中提取SuperCLEVR特定信息
        self.question_type = self.metadata.get('question_type', 'general')
        self.expected_answer = self.answer.lower() if self.answer else None
        self.is_binary = self.metadata.get('answer_type', 'binary') == 'binary'
        
        # 获取问题分析信息
        self.question_analysis = self.metadata.get('question_analysis', {})
        self.attributes = self.question_analysis.get('attributes', {})
        self.complexity_features = self.question_analysis.get('complexity', {})
        self.is_complex = self.question_analysis.get('is_complex', False)
        
        # 分析问题特征
        question_features = self._analyze_question_features()
        
        # 添加SuperCLEVR特定的信息
        task_info["task_type"] = self.task_type
        task_info["question_type"] = self.question_type
        task_info["expected_answer"] = self.expected_answer
        task_info["is_binary"] = self.is_binary
        task_info["is_complex"] = self.is_complex
        task_info["question_features"] = question_features
        task_info["object_attributes"] = self.attributes
        task_info["complexity_features"] = self.complexity_features
        task_info["dataset"] = "superclevr"
        task_info["difficulty"] = self._assess_difficulty()
        
        # 修改任务目标以包含SuperCLEVR特定指导
        enhanced_goal = self._enhance_task_goal(task_goal, question_features)
        
        return enhanced_goal, task_info
    
    def _enhance_task_goal(self, base_goal: str, features: Dict[str, bool]) -> str:
        """增强任务目标描述，添加SuperCLEVR特定的指导"""
        enhanced_parts = [base_goal]
        
        # 添加CLEVR场景理解通用指导
        enhanced_parts.append("\n**CLEVR Scene Understanding:**")
        enhanced_parts.append("1. Carefully observe all objects in the synthetic scene")
        enhanced_parts.append("2. Identify object properties: color, shape, size, material")
        enhanced_parts.append("3. Note spatial relationships between objects")
        enhanced_parts.append("4. Focus on the specific aspects the question asks about")
        
        # 根据任务类型添加特定指导
        if self.task_type == 'visual_comparison':
            enhanced_parts.append("\n**Visual Comparison Task:**")
            enhanced_parts.append("- Compare the specified attributes (size, color, shape, material)")
            enhanced_parts.append("- Determine if they are the SAME or DIFFERENT")
            enhanced_parts.append("- Answer 'yes' if same, 'no' if different")
            if 'same size' in self.question.lower():
                enhanced_parts.append("- Focus specifically on comparing sizes")
            elif 'same color' in self.question.lower():
                enhanced_parts.append("- Focus specifically on comparing colors")
        
        elif self.task_type == 'counting_comparison':
            enhanced_parts.append("\n**Counting Comparison Task:**")
            enhanced_parts.append("- Count objects in each specified category")
            enhanced_parts.append("- Compare if the numbers are EQUAL")
            enhanced_parts.append("- Answer 'yes' if equal, 'no' if not equal")
            enhanced_parts.append("- Be precise about which objects to count")
        
        elif self.task_type == 'object_counting':
            enhanced_parts.append("\n**Object Counting Task:**")
            enhanced_parts.append("- Count ONLY objects matching ALL specified criteria")
            enhanced_parts.append("- Be careful about attribute combinations")
            enhanced_parts.append("- The answer should be a specific number")
        
        elif self.task_type == 'object_existence':
            enhanced_parts.append("\n**Existence Check Task:**")
            enhanced_parts.append("- Check if ANY object matches the criteria")
            enhanced_parts.append("- Answer 'yes' if at least one exists")
            enhanced_parts.append("- Answer 'no' if none exist")
        
        elif self.task_type == 'spatial_reasoning':
            enhanced_parts.append("\n**Spatial Reasoning Task:**")
            enhanced_parts.append("- Identify the reference object first")
            enhanced_parts.append("- Check spatial relationships (left/right/behind/front)")
            enhanced_parts.append("- Consider the viewpoint/perspective")
            if features.get('has_left_right'):
                enhanced_parts.append("- Left/right is from the viewer's perspective")
        
        # 复杂问题警告
        if self.is_complex:
            enhanced_parts.append("\n**⚠️ Complex Question Warning:**")
            enhanced_parts.append("- This question has multiple conditions")
            enhanced_parts.append("- Check ALL conditions carefully")
            if features.get('has_and'):
                enhanced_parts.append("- ALL conditions must be true for 'yes'")
            if features.get('has_or'):
                enhanced_parts.append("- ANY condition being true means 'yes'")
            if features.get('has_negation'):
                enhanced_parts.append("- Watch for negations (not, n't)")
        
        # 物体属性提示
        if self.attributes:
            enhanced_parts.append("\n**Objects to Look For:**")
            if self.attributes.get('colors'):
                enhanced_parts.append(f"- Colors: {', '.join(self.attributes['colors'])}")
            if self.attributes.get('shapes'):
                enhanced_parts.append(f"- Shapes: {', '.join(self.attributes['shapes'])}")
            if self.attributes.get('sizes'):
                enhanced_parts.append(f"- Sizes: {', '.join(self.attributes['sizes'])}")
            if self.attributes.get('materials'):
                enhanced_parts.append(f"- Materials: {', '.join(self.attributes['materials'])}")
            if self.attributes.get('objects'):
                enhanced_parts.append(f"- Objects: {', '.join(self.attributes['objects'])}")
        
        # 答案格式指导
        enhanced_parts.append("\n**Answer Format:**")
        if self.is_binary:
            enhanced_parts.append("- Answer ONLY 'yes' or 'no'")
            enhanced_parts.append("- No explanation needed, just the answer")
        else:
            enhanced_parts.append("- Provide a single answer")
            enhanced_parts.append("- For counting, give just the number")
        
        # 常见错误提醒
        enhanced_parts.append("\n**Common Mistakes to Avoid:**")
        if self.task_type == 'visual_comparison':
            enhanced_parts.append("- Don't confuse different attribute types")
            enhanced_parts.append("- 'Same' means identical, not similar")
        elif self.task_type == 'counting_comparison':
            enhanced_parts.append("- Count each group separately before comparing")
            enhanced_parts.append("- Include ALL objects matching the criteria")
        elif self.task_type == 'spatial_reasoning':
            enhanced_parts.append("- Consider the correct reference frame")
            enhanced_parts.append("- 'Left of X' means to the left side of object X")
        
        return "\n".join(enhanced_parts)
    
    def _analyze_question_features(self) -> Dict[str, bool]:
        """分析问题特征"""
        if not self.question:
            return {}
        
        q_lower = self.question.lower()
        
        features = {
            'has_comparison': any(word in q_lower for word in ['same', 'equal', 'more', 'less', 'fewer']),
            'has_counting': 'how many' in q_lower or 'number of' in q_lower,
            'has_existence': any(phrase in q_lower for phrase in ['are there', 'is there', 'exist', 'any']),
            'has_spatial': any(word in q_lower for word in self.CLEVR_SPATIAL),
            'has_left_right': 'left' in q_lower or 'right' in q_lower,
            'has_and': ' and ' in q_lower,
            'has_or': ' or ' in q_lower,
            'has_negation': any(word in q_lower for word in ['not', "n't", 'no ']),
            'has_all': 'all' in q_lower,
            'has_any': 'any' in q_lower
        }
        
        # Add specific comparison types
        features['has_size_comparison'] = 'same size' in q_lower
        features['has_color_comparison'] = 'same color' in q_lower
        features['has_shape_comparison'] = 'same shape' in q_lower
        features['has_material_comparison'] = 'same material' in q_lower
        
        return features
    
    def check_success(self, action: Any) -> Tuple[bool, str]:
        """
        检查SuperCLEVR答案
        
        主要检查yes/no答案
        """
        if action is None:
            return False, "No answer provided"
        
        # 规范化用户答案
        user_answer = self._normalize_answer(action)
        
        if user_answer is None:
            if self.is_binary:
                return False, "Please answer 'yes' or 'no'"
            else:
                return False, "Please provide a valid answer"
        
        # 获取正确答案
        correct_answer = self.expected_answer
        
        # 检查是否匹配
        if user_answer == correct_answer:
            return True, f"Correct! The answer is '{correct_answer}'"
        else:
            # 提供有用的反馈
            if self.is_binary:
                explanation = self._get_error_explanation(user_answer, correct_answer)
                return False, f"Incorrect. The answer is '{correct_answer}'. {explanation}"
            else:
                return False, f"Incorrect. The answer is '{correct_answer}' (you said '{user_answer}')"
    
    def _normalize_answer(self, answer: Any) -> Optional[str]:
        """规范化答案格式"""
        if answer is None:
            return None
        
        answer_str = str(answer).strip().lower()
        
        # 对于二元答案
        if self.is_binary:
            # 处理常见的yes变体
            if answer_str in ['yes', 'y', 'true', '1', 'correct', 'right', 'affirmative']:
                return 'yes'
            # 处理常见的no变体
            elif answer_str in ['no', 'n', 'false', '0', 'incorrect', 'wrong', 'negative']:
                return 'no'
            # 尝试从句子中提取yes/no
            elif 'yes' in answer_str and 'no' not in answer_str:
                return 'yes'
            elif 'no' in answer_str and 'yes' not in answer_str:
                return 'no'
            else:
                return None
        else:
            # 对于非二元答案，直接返回
            return answer_str
    
    def _get_error_explanation(self, user_answer: str, correct_answer: str) -> str:
        """生成错误解释"""
        explanations = []
        
        if self.task_type == 'visual_comparison':
            if correct_answer == 'yes':
                explanations.append("The objects DO have the same attribute.")
            else:
                explanations.append("The objects do NOT have the same attribute.")
        
        elif self.task_type == 'counting_comparison':
            if correct_answer == 'yes':
                explanations.append("The numbers ARE equal.")
            else:
                explanations.append("The numbers are NOT equal.")
        
        elif self.task_type == 'object_existence':
            if correct_answer == 'yes':
                explanations.append("Such objects DO exist in the scene.")
            else:
                explanations.append("No such objects exist in the scene.")
        
        elif self.task_type == 'spatial_reasoning':
            if correct_answer == 'yes':
                explanations.append("The spatial relationship IS correct.")
            else:
                explanations.append("The spatial relationship is NOT correct.")
        
        if self.is_complex:
            if self.complexity_features.get('has_and'):
                explanations.append("Remember: ALL conditions must be satisfied.")
            if self.complexity_features.get('has_negation'):
                explanations.append("Note the negation in the question.")
        
        return " ".join(explanations) if explanations else ""
    
    def validate(
        self,
        chat_history: List[Dict],
        observation: Any,
        full_history: Optional[List[Any]] = None
    ) -> Tuple[float, bool, str, Dict[str, Any]]:
        """
        验证SuperCLEVR任务执行情况
        """
        # 调用父类验证
        reward, done, message, info = super().validate(
            chat_history, observation, full_history
        )
        
        # 添加SuperCLEVR特定的信息
        info["task_type"] = self.task_type
        info["question_type"] = self.question_type
        info["expected_answer"] = self.expected_answer
        info["is_binary"] = self.is_binary
        info["is_complex"] = self.is_complex
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
            "user_answer": None,
            "correct_answer": self.expected_answer,
            "normalized_user_answer": None
        }
        
        # 规范化用户答案
        normalized = self._normalize_answer(user_answer)
        error_info["user_answer"] = str(user_answer)
        error_info["normalized_user_answer"] = normalized
        
        if normalized is None:
            error_info["error_type"] = "invalid_format"
        elif self.is_binary and normalized != self.expected_answer:
            # 分析二元错误
            if self.task_type == 'visual_comparison':
                error_info["error_type"] = "attribute_confusion"
            elif self.task_type == 'counting_comparison':
                error_info["error_type"] = "counting_error"
            elif self.task_type == 'object_existence':
                error_info["error_type"] = "detection_error"
            elif self.task_type == 'spatial_reasoning':
                error_info["error_type"] = "spatial_confusion"
            else:
                error_info["error_type"] = "logic_error"
            
            # 复杂问题的特殊错误
            if self.is_complex:
                if self.complexity_features.get('has_negation'):
                    error_info["error_type"] = "negation_misunderstanding"
                elif self.complexity_features.get('has_and'):
                    error_info["error_type"] = "conjunction_error"
        
        return error_info
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取SuperCLEVR特定的指标"""
        metrics = super().get_metrics()
        
        metrics.update({
            "task_type": self.task_type,
            "question_type": self.question_type,
            "expected_answer": self.expected_answer,
            "is_binary": self.is_binary,
            "is_complex": self.is_complex,
            "difficulty": self._assess_difficulty(),
            "question_features": self._analyze_question_features(),
            "object_attributes": self.attributes,
            "complexity_features": self.complexity_features
        })
        
        return metrics
    
    def _assess_difficulty(self) -> str:
        """评估任务难度"""
        difficulty_score = 0
        
        # 基于复杂度
        if self.is_complex:
            difficulty_score += 2
        
        # 基于复杂度特征数量
        complexity_count = sum(1 for v in self.complexity_features.values() if v)
        difficulty_score += complexity_count
        
        # 基于任务类型
        if self.task_type == 'spatial_reasoning':
            difficulty_score += 1
        elif self.task_type == 'counting_comparison':
            difficulty_score += 1
        
        # 基于属性数量
        total_attributes = sum(len(v) for v in self.attributes.values() if isinstance(v, list))
        if total_attributes > 3:
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
        
        # 添加SuperCLEVR特定信息
        obs["scene_type"] = "CLEVR synthetic scene"
        obs["expected_objects"] = "geometric shapes with various colors, sizes, and materials"
        obs["task_focus"] = "visual reasoning and logical judgment"
        obs["answer_type"] = "binary (yes/no)" if self.is_binary else "open-ended"
        
        return obs
    
    def get_task_description(self) -> str:
        """获取任务描述"""
        desc_parts = [
            f"This is a visual reasoning task on a CLEVR synthetic image.",
            f"Task type: {self.TASK_TYPES.get(self.task_type, self.task_type)}",
            f"Question type: {self.QUESTION_TYPES.get(self.question_type, self.question_type)}"
        ]
        
        if self.is_complex:
            desc_parts.append("⚠️ This is a complex question with multiple conditions.")
        
        if self.is_binary:
            desc_parts.append(f"Expected answer: '{self.expected_answer}' (yes/no)")
        else:
            desc_parts.append(f"Expected answer: '{self.expected_answer}'")
        
        desc_parts.append(f"Difficulty: {self._assess_difficulty()}")
        
        return "\n".join(desc_parts)