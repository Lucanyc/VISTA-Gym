#!/usr/bin/env python3
"""
VQA-COCO Task implementation for VLM Gym
Handles visual question answering on natural images from COCO dataset
"""

from typing import Tuple, Dict, Any, List, Optional
import re
from collections import Counter

from .vision_qa_task import VisionQATask


class VQACOCOTask(VisionQATask):
    """
    VQA-COCO 特定任务
    
    专门处理COCO图像的视觉问答任务，支持多种问题类型：
    - binary: Yes/No questions
    - counting: Questions about quantity
    - color: Questions about colors
    - object: Object identification questions
    - location: Questions about position/location
    - activity: Questions about actions/activities
    - general: Other open-ended questions
    
    使用VQA特定的评分机制：
    - accuracy = min(#humans that provided that answer / 3, 1)
    """
    
    # VQA-COCO的任务类型（基于问题内容分类）
    TASK_TYPES = {
        'vqa_binary': 'Yes/No questions',
        'vqa_counting': 'Counting objects or quantity questions',
        'vqa_color': 'Questions about colors',
        'vqa_object': 'Object identification questions',
        'vqa_location': 'Questions about position or location',
        'vqa_activity': 'Questions about actions or activities',
        'vqa_general': 'General open-ended questions'
    }
    
    # VQA答案类型（来自原始数据集）
    ANSWER_TYPES = {
        'yes/no': 'Binary yes or no answers',
        'number': 'Numeric answers',
        'other': 'Other types of answers'
    }
    
    # 常见的答案标准化映射
    ANSWER_NORMALIZATIONS = {
        # Numbers
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12',
        # Yes/No variations
        'yep': 'yes', 'yeah': 'yes', 'yup': 'yes', 'correct': 'yes',
        'nope': 'no', 'nah': 'no', 'incorrect': 'no',
        # Common phrases
        "don't know": 'unknown', 'not sure': 'unknown', 'cant tell': 'unknown'
    }
    
    @classmethod
    def get_task_id(cls) -> str:
        """获取任务类型ID"""
        return "vlm-gym.vqacoco"
    
    def setup(self) -> Tuple[str, Dict[str, Any]]:
        """设置VQA-COCO特定的任务"""
        # 调用父类setup
        task_goal, task_info = super().setup()
        
        # 从metadata中提取VQA特定信息
        self.question_id = self.metadata.get('question_id', -1)
        self.image_id = self.metadata.get('image_id', -1)
        self.image_name = self.metadata.get('image_name', '')
        self.answer_type = self.metadata.get('answer_type', 'other')
        self.question_type = self.metadata.get('question_type', 'unknown')
        self.task_type = task_info.get('task', 'vqa_general')
        
        # VQA特有的多答案信息
        self.all_answers = self.metadata.get('all_answers', [])
        self.answer_frequencies = self.metadata.get('answer_frequencies', {})
        self.multiple_choice_answer = self.metadata.get('multiple_choice_answer', '')
        
        # 分析问题特征
        self.is_binary = self.task_type == 'vqa_binary' or self.answer_type == 'yes/no'
        self.is_counting = self.task_type == 'vqa_counting'
        self.is_color = self.task_type == 'vqa_color'
        self.requires_object_detection = self._check_requires_detection()
        self.requires_spatial_reasoning = self._check_requires_spatial()
        
        # 计算答案的置信度（基于人类一致性）
        self.answer_confidence = self._calculate_answer_confidence()
        
        # 添加VQA特定信息到task_info
        task_info["question_id"] = self.question_id
        task_info["image_id"] = self.image_id
        task_info["image_name"] = self.image_name
        task_info["answer_type"] = self.answer_type
        task_info["question_type"] = self.question_type
        task_info["task_type"] = self.task_type
        task_info["is_binary"] = self.is_binary
        task_info["is_counting"] = self.is_counting
        task_info["is_color"] = self.is_color
        task_info["answer_confidence"] = self.answer_confidence
        task_info["human_agreement"] = self._get_human_agreement()
        task_info["dataset"] = "vqa_coco"
        
        # 修改任务目标以包含VQA特定指导
        enhanced_goal = self._enhance_task_goal(task_goal)
        
        return enhanced_goal, task_info
    
    def _enhance_task_goal(self, base_goal: str) -> str:
        """增强任务目标描述，添加VQA特定的指导"""
        enhanced_parts = [base_goal]
        
        # 根据任务类型添加特定指导
        if self.is_binary:
            enhanced_parts.append("\n**This is a Yes/No question.**")
            enhanced_parts.append("- Look carefully at the image to determine if the statement is true or false")
            enhanced_parts.append("- Answer with a clear 'Yes' or 'No'")
            
        elif self.is_counting:
            enhanced_parts.append("\n**This is a counting question.**")
            enhanced_parts.append("- Count all visible instances of the requested object/entity")
            enhanced_parts.append("- Be careful not to miss partially visible items")
            enhanced_parts.append("- Provide a specific number as your answer")
            
        elif self.is_color:
            enhanced_parts.append("\n**This is a color-related question.**")
            enhanced_parts.append("- Identify the color(s) being asked about")
            enhanced_parts.append("- Consider lighting conditions that might affect color appearance")
            enhanced_parts.append("- Use common color names in your answer")
            
        elif self.task_type == 'vqa_object':
            enhanced_parts.append("\n**This is an object identification question.**")
            enhanced_parts.append("- Identify the specific object or entity being asked about")
            enhanced_parts.append("- Be specific but use common terms")
            
        elif self.task_type == 'vqa_location':
            enhanced_parts.append("\n**This is a location/position question.**")
            enhanced_parts.append("- Describe positions using clear spatial terms (left, right, top, bottom, etc.)")
            enhanced_parts.append("- Consider the perspective from which the image is taken")
            
        elif self.task_type == 'vqa_activity':
            enhanced_parts.append("\n**This is an activity/action question.**")
            enhanced_parts.append("- Focus on what the subjects are doing")
            enhanced_parts.append("- Use present continuous tense for ongoing actions")
        
        # 添加关于答案置信度的信息
        if self.answer_confidence < 0.5:
            enhanced_parts.append("\n**Note**: This question has low human agreement, suggesting it may be ambiguous.")
            enhanced_parts.append("Provide your best interpretation based on what you observe.")
        
        # 添加通用的VQA提示
        enhanced_parts.append("\n**General VQA Tips:**")
        enhanced_parts.append("- Focus on the specific aspect the question asks about")
        enhanced_parts.append("- Be concise but accurate in your answer")
        enhanced_parts.append("- Base your answer only on what's visible in the image")
        
        return "\n".join(enhanced_parts)
    
    def _calculate_answer_confidence(self) -> float:
        """计算答案的置信度（基于人类标注者的一致性）"""
        if not self.answer_frequencies:
            return 0.0
        
        total_annotations = sum(self.answer_frequencies.values())
        if total_annotations == 0:
            return 0.0
        
        # 获取最常见答案的频率
        max_frequency = max(self.answer_frequencies.values())
        confidence = max_frequency / total_annotations
        
        return confidence
    
    def _get_human_agreement(self) -> float:
        """获取人类标注者的一致性程度"""
        if not self.answer_frequencies:
            return 0.0
        
        # 计算最常见答案占比
        max_freq = max(self.answer_frequencies.values()) if self.answer_frequencies else 0
        return max_freq / 10.0  # VQA有10个标注者
    
    def _check_requires_detection(self) -> bool:
        """检查是否需要物体检测"""
        if not self.question:
            return False
        
        question_lower = self.question.lower()
        detection_keywords = ['what is', 'what are', 'what kind', 'what type', 
                            'identify', 'name the', 'which object']
        return any(keyword in question_lower for keyword in detection_keywords)
    
    def _check_requires_spatial(self) -> bool:
        """检查是否需要空间推理"""
        if not self.question:
            return False
        
        question_lower = self.question.lower()
        spatial_keywords = ['where', 'position', 'location', 'left', 'right', 
                          'above', 'below', 'between', 'next to', 'behind', 'front']
        return any(keyword in question_lower for keyword in spatial_keywords)
    
    def check_success(self, action: Any) -> Tuple[bool, str]:
        """
        检查VQA答案使用VQA特定的评分机制
        
        VQA评分: accuracy = min(#humans that provided that answer / 3, 1)
        """
        if not action:
            return False, "No answer provided"
        
        # 标准化用户答案
        user_answer = self._normalize_answer(str(action).strip())
        
        # 如果没有答案频率信息，使用简单匹配
        if not self.answer_frequencies:
            correct_answer = self._normalize_answer(str(self.answer).strip())
            if user_answer == correct_answer:
                return True, f"Correct! The answer is {self.answer}"
            else:
                return False, f"Incorrect. The expected answer is {self.answer}"
        
        # 使用VQA评分机制
        human_count = 0
        for human_answer, count in self.answer_frequencies.items():
            if self._normalize_answer(human_answer) == user_answer:
                human_count = count
                break
        
        # 计算VQA分数
        vqa_score = min(human_count / 3.0, 1.0)
        
        # 决定是否正确（阈值可调整）
        if vqa_score >= 1.0:
            return True, f"Correct! {human_count}/10 human annotators gave the same answer: {action}"
        elif vqa_score >= 0.67:  # 至少2个人同意
            return True, f"Acceptable answer! {human_count}/10 human annotators agreed: {action}"
        elif vqa_score > 0:
            return False, f"Partially correct. Only {human_count}/10 annotators gave this answer. Most common answer: {self.answer}"
        else:
            # 提供最常见的几个答案作为参考
            top_answers = sorted(self.answer_frequencies.items(), key=lambda x: x[1], reverse=True)[:3]
            answer_str = ", ".join([f"{ans} ({cnt})" for ans, cnt in top_answers])
            return False, f"Incorrect. Common answers from annotators: {answer_str}"
    
    def _normalize_answer(self, answer: str) -> str:
        """标准化答案以便比较"""
        # 转小写
        normalized = answer.lower().strip()
        
        # 移除标点符号
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        # 移除冠词
        normalized = re.sub(r'\b(a|an|the)\b', '', normalized)
        
        # 应用常见的标准化映射
        for pattern, replacement in self.ANSWER_NORMALIZATIONS.items():
            if normalized == pattern:
                normalized = replacement
                break
        
        # 移除多余空格
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def validate(
        self,
        chat_history: List[Dict],
        observation: Any,
        full_history: Optional[List[Any]] = None
    ) -> Tuple[float, bool, str, Dict[str, Any]]:
        """
        验证VQA-COCO任务执行情况
        """
        # 调用父类验证
        reward, done, message, info = super().validate(
            chat_history, observation, full_history
        )
        
        # 添加VQA特定的信息
        info["question_id"] = self.question_id
        info["image_id"] = self.image_id
        info["answer_type"] = self.answer_type
        info["question_type"] = self.question_type
        info["task_type"] = self.task_type
        info["is_binary"] = self.is_binary
        info["is_counting"] = self.is_counting
        info["is_color"] = self.is_color
        info["answer_confidence"] = self.answer_confidence
        info["human_agreement"] = self._get_human_agreement()
        
        # 如果提供了答案，计算VQA分数
        if info.get("answer_provided"):
            provided_answer = str(info["answer_provided"])
            normalized_answer = self._normalize_answer(provided_answer)
            
            # 计算VQA分数
            human_count = 0
            for human_answer, count in self.answer_frequencies.items():
                if self._normalize_answer(human_answer) == normalized_answer:
                    human_count = count
                    break
            
            vqa_score = min(human_count / 3.0, 1.0) if self.answer_frequencies else 0.0
            info["vqa_score"] = vqa_score
            info["human_count"] = human_count
            
            # 检查答案是否在人类答案中
            info["answer_in_human_answers"] = normalized_answer in [
                self._normalize_answer(ans) for ans in self.all_answers
            ]
            
            # 对于yes/no问题，检查极性是否正确
            if self.is_binary:
                correct_polarity = self._normalize_answer(self.answer) in ['yes', 'no']
                user_polarity = normalized_answer in ['yes', 'no']
                info["correct_polarity"] = correct_polarity and user_polarity
        
        return reward, done, message, info
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取VQA-COCO特定的指标"""
        metrics = super().get_metrics()
        
        metrics.update({
            "question_id": self.question_id,
            "image_id": self.image_id,
            "image_name": self.image_name,
            "answer_type": self.answer_type,
            "question_type": self.question_type,
            "task_type": self.task_type,
            "is_binary": self.is_binary,
            "is_counting": self.is_counting,
            "is_color": self.is_color,
            "answer_confidence": self.answer_confidence,
            "human_agreement": self._get_human_agreement(),
            "unique_answers": len(self.answer_frequencies),
            "requires_object_detection": self.requires_object_detection,
            "requires_spatial_reasoning": self.requires_spatial_reasoning,
            "difficulty_factors": self.get_difficulty_factors()
        })
        
        return metrics
    
    def get_difficulty_factors(self) -> List[str]:
        """获取影响难度的因素"""
        factors = []
        
        # 基于人类一致性
        if self.answer_confidence < 0.3:
            factors.append("high_ambiguity")
        elif self.answer_confidence < 0.5:
            factors.append("moderate_ambiguity")
        
        # 基于答案多样性
        if len(self.answer_frequencies) > 5:
            factors.append("diverse_answers")
        
        # 基于问题类型
        if self.is_counting:
            factors.append("requires_counting")
        
        if self.requires_object_detection:
            factors.append("requires_detection")
        
        if self.requires_spatial_reasoning:
            factors.append("requires_spatial_reasoning")
        
        # 基于问题长度
        if self.question and len(self.question.split()) > 15:
            factors.append("complex_question")
        
        # 基于答案类型
        if self.answer_type == 'other' and not self.is_binary:
            factors.append("open_ended")
        
        return factors
    
    def get_human_answers_distribution(self) -> Dict[str, int]:
        """获取人类答案的分布（辅助方法）"""
        return self.answer_frequencies
    
    def get_answer_diversity(self) -> float:
        """计算答案的多样性（熵）"""
        if not self.answer_frequencies:
            return 0.0
        
        total = sum(self.answer_frequencies.values())
        if total == 0:
            return 0.0
        
        # 计算熵
        import math
        entropy = 0.0
        for count in self.answer_frequencies.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        return entropy
    
    def get_hint(self) -> str:
        """获取问题提示（用于辅助）"""
        hints = []
        
        if self.is_binary:
            hints.append("This is a yes/no question. Look for evidence in the image.")
        elif self.is_counting:
            hints.append("Count carefully, including partially visible objects.")
        elif self.is_color:
            hints.append("Focus on the colors visible in the relevant parts of the image.")
        elif self.task_type == 'vqa_object':
            hints.append("Identify the main object or entity being asked about.")
        elif self.task_type == 'vqa_location':
            hints.append("Pay attention to spatial relationships and positions.")
        elif self.task_type == 'vqa_activity':
            hints.append("Focus on what actions or activities are taking place.")
        
        # 如果答案置信度低，提醒可能有多种解释
        if self.answer_confidence < 0.5:
            hints.append("Note: This question may have multiple valid interpretations.")
        
        return " ".join(hints) if hints else "Analyze the image carefully to answer the question."