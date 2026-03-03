"""
VQA-AS Task implementation for VLM Gym
Handles Visual Question Answering tasks on COCO images with diverse question types
"""

from typing import Tuple, Dict, Any, List, Optional
import re
import json
from pathlib import Path
from collections import Counter

from .vision_qa_task import VisionQATask


class VqaAsTask(VisionQATask):
    """
    VQA-AS (Visual Question Answering - Answer Set) 特定任务
    
    专门处理基于COCO真实图像的视觉问答任务，包括：
    - 物体识别（what is, what type）
    - 颜色识别（what color）
    - 数量计数（how many）
    - 位置推理（where）
    - 活动识别（what is X doing）
    - Yes/No验证问题
    - 属性识别（what kind, which）
    - 时间推理（when, what time）
    - 人物识别（who）
    
    支持二元判断和开放式答案
    """
    
    # 任务类型
    TASK_TYPES = {
        'object_recognition': '物体识别',
        'color_recognition': '颜色识别',
        'object_counting': '物体计数',
        'spatial_reasoning': '空间位置推理',
        'activity_recognition': '活动识别',
        'visual_verification': '视觉验证（Yes/No）',
        'attribute_recognition': '属性识别',
        'temporal_reasoning': '时间推理',
        'person_recognition': '人物识别',
        'visual_reasoning': '视觉推理',
        'visual_qa': '通用视觉问答'
    }
    
    # 问题类型
    QUESTION_TYPES = {
        'identification': '识别类（what is/are）',
        'color': '颜色类',
        'counting': '计数类',
        'location': '位置类',
        'activity': '活动类',
        'yes_no': 'Yes/No判断类',
        'attribute': '属性类',
        'time': '时间类',
        'person': '人物类',
        'reasoning': '推理类',
        'general': '通用类'
    }
    
    # 常见答案类别
    ANSWER_CATEGORIES = {
        'yes_no': ['yes', 'no'],
        'colors': ['red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'gray', 'orange', 'pink', 'purple'],
        'numbers': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'one', 'two', 'three', 'four', 'five'],
        'activities': ['sitting', 'standing', 'walking', 'running', 'eating', 'drinking', 'playing', 'reading', 'sleeping', 'talking'],
        'locations': ['table', 'chair', 'floor', 'ground', 'wall', 'sky', 'street', 'room', 'building', 'outside', 'inside']
    }
    
    def __init__(self, task_id: str, adapter: Any):
        """
        初始化VQA-AS任务
        
        Args:
            task_id: 任务ID
            adapter: VQA数据适配器
        """
        # 调用父类初始化
        super().__init__(task_id, adapter)
        
        # 初始化VQA特定属性
        self.task_type = None
        self.question_type = None
        self.answer_type = None  # 'yes_no' or 'open_ended'
        self.expected_answer = None
        self.answer_distribution = {}
        self.question_analysis = {}
        self.entities = {}
        self.complexity_features = {}
        self.is_complex = False
        self.is_binary = False
        self.answer_length_category = None  # 'short', 'medium', 'long'
        
        # 获取任务数据
        task_data = adapter.get_task_data(task_id)
        self.dataset_name = task_data.get('dataset', 'vqa')
        self.task_type = task_data.get('task', 'visual_qa')
        self.choices = task_data.get('choices', None)
    
    @classmethod
    def get_task_id(cls) -> str:
        """获取任务类型ID"""
        return "vlm-gym.vqa-as"
    
    def setup(self) -> Tuple[str, Dict[str, Any]]:
        """设置VQA-AS特定的任务"""
        # 调用父类setup
        task_goal, task_info = super().setup()
        
        # 从metadata中提取VQA特定信息
        self.question_type = self.metadata.get('question_type', 'general')
        self.answer_type = self.metadata.get('answer_type', 'open_ended')
        self.expected_answer = self.answer.lower() if self.answer else None
        self.is_binary = self.answer_type == 'yes_no' or self.expected_answer in ['yes', 'no']
        
        # 获取答案分布
        self.answer_distribution = self.metadata.get('answer_distribution', {})
        
        # 获取问题分析信息
        self.question_analysis = self.metadata.get('question_analysis', {})
        self.entities = self.question_analysis.get('entities', {})
        self.complexity_features = self.question_analysis.get('complexity', {})
        self.is_complex = self.question_analysis.get('is_complex', False)
        
        # 判断答案长度类别
        answer_words = len(self.expected_answer.split()) if self.expected_answer else 0
        if answer_words == 1:
            self.answer_length_category = 'short'
        elif answer_words <= 3:
            self.answer_length_category = 'medium'
        else:
            self.answer_length_category = 'long'
        
        # 分析问题特征
        question_features = self._analyze_question_features()
        
        # 添加VQA特定的信息
        task_info["task_type"] = self.task_type
        task_info["question_type"] = self.question_type
        task_info["answer_type"] = self.answer_type
        task_info["expected_answer"] = self.expected_answer
        task_info["is_binary"] = self.is_binary
        task_info["is_complex"] = self.is_complex
        task_info["answer_length"] = self.answer_length_category
        task_info["answer_distribution"] = self.answer_distribution
        task_info["num_unique_answers"] = len(self.answer_distribution)
        task_info["question_features"] = question_features
        task_info["entities"] = self.entities
        task_info["complexity_features"] = self.complexity_features
        task_info["dataset"] = "vqa"
        task_info["difficulty"] = self._assess_difficulty()
        task_info["answer_consensus"] = self._calculate_answer_consensus()
        
        # 修改任务目标以包含VQA特定指导
        enhanced_goal = self._enhance_task_goal(task_goal, question_features)
        
        return enhanced_goal, task_info
    
    def _enhance_task_goal(self, base_goal: str, features: Dict[str, bool]) -> str:
        """增强任务目标描述，添加VQA特定的指导"""
        enhanced_parts = [base_goal]
        
        # 添加VQA场景理解通用指导
        enhanced_parts.append("\n**VQA Scene Understanding:**")
        enhanced_parts.append("1. Carefully observe the real-world image")
        enhanced_parts.append("2. Identify relevant objects, people, and activities")
        enhanced_parts.append("3. Consider the context and relationships")
        enhanced_parts.append("4. Focus on answering the specific question asked")
        
        # 根据任务类型添加特定指导
        if self.task_type == 'object_recognition':
            enhanced_parts.append("\n**Object Recognition Task:**")
            enhanced_parts.append("- Identify what objects are present")
            enhanced_parts.append("- Be specific about the object type")
            enhanced_parts.append("- Consider the main subject of the question")
        
        elif self.task_type == 'color_recognition':
            enhanced_parts.append("\n**Color Recognition Task:**")
            enhanced_parts.append("- Focus on the color of the specified object")
            enhanced_parts.append("- Use common color names")
            enhanced_parts.append("- Be precise (e.g., 'red' not 'reddish')")
        
        elif self.task_type == 'object_counting':
            enhanced_parts.append("\n**Counting Task:**")
            enhanced_parts.append("- Count ALL visible instances")
            enhanced_parts.append("- Be careful not to miss partially visible objects")
            enhanced_parts.append("- Provide a specific number")
        
        elif self.task_type == 'spatial_reasoning':
            enhanced_parts.append("\n**Spatial/Location Task:**")
            enhanced_parts.append("- Identify the location or spatial relationships")
            enhanced_parts.append("- Use clear spatial terms")
            enhanced_parts.append("- Consider the perspective")
        
        elif self.task_type == 'activity_recognition':
            enhanced_parts.append("\n**Activity Recognition Task:**")
            enhanced_parts.append("- Identify what action is being performed")
            enhanced_parts.append("- Focus on the main activity")
            enhanced_parts.append("- Use present continuous tense if applicable")
        
        elif self.task_type == 'visual_verification':
            enhanced_parts.append("\n**Yes/No Verification Task:**")
            enhanced_parts.append("- Carefully verify the statement in the question")
            enhanced_parts.append("- Answer only 'yes' or 'no'")
            enhanced_parts.append("- Consider all aspects before deciding")
        
        # 实体提示
        if self.entities:
            enhanced_parts.append("\n**Relevant Entities:**")
            if self.entities.get('colors'):
                enhanced_parts.append(f"- Colors mentioned: {', '.join(self.entities['colors'])}")
            if self.entities.get('activities'):
                enhanced_parts.append(f"- Activities mentioned: {', '.join(self.entities['activities'])}")
            if self.entities.get('locations'):
                enhanced_parts.append(f"- Locations mentioned: {', '.join(self.entities['locations'])}")
        
        # 答案格式指导
        enhanced_parts.append("\n**Answer Format:**")
        if self.is_binary:
            enhanced_parts.append("- Answer ONLY 'yes' or 'no'")
            enhanced_parts.append("- No explanation needed")
        else:
            enhanced_parts.append(f"- Expected answer length: {self.answer_length_category}")
            if self.answer_length_category == 'short':
                enhanced_parts.append("- Provide a concise 1-word answer")
            elif self.answer_length_category == 'medium':
                enhanced_parts.append("- Provide a brief 2-3 word answer")
            else:
                enhanced_parts.append("- Provide a complete but concise answer")
        
        # 答案分布提示（如果有多个常见答案）
        if len(self.answer_distribution) > 1:
            top_answers = sorted(self.answer_distribution.items(), key=lambda x: x[1], reverse=True)[:3]
            enhanced_parts.append(f"\n**Common answers for similar questions:** {', '.join([ans for ans, _ in top_answers])}")
        
        # 复杂问题警告
        if self.is_complex:
            enhanced_parts.append("\n**⚠️ Complex Question Warning:**")
            enhanced_parts.append("- This question has multiple aspects to consider")
            if features.get('has_spatial'):
                enhanced_parts.append("- Pay attention to spatial relationships")
            if features.get('has_comparison'):
                enhanced_parts.append("- Compare different aspects carefully")
        
        # 常见错误提醒
        enhanced_parts.append("\n**Common Mistakes to Avoid:**")
        if self.task_type == 'object_counting':
            enhanced_parts.append("- Don't forget to count all instances")
            enhanced_parts.append("- Include partially visible objects")
        elif self.task_type == 'color_recognition':
            enhanced_parts.append("- Focus on the specific object asked about")
            enhanced_parts.append("- Don't confuse similar colors")
        elif self.task_type == 'activity_recognition':
            enhanced_parts.append("- Describe what IS happening, not what might happen")
            enhanced_parts.append("- Be specific about the action")
        
        return "\n".join(enhanced_parts)
    
    def _analyze_question_features(self) -> Dict[str, bool]:
        """分析问题特征"""
        if not self.question:
            return {}
        
        q_lower = self.question.lower()
        
        features = {
            'starts_with_what': q_lower.startswith('what'),
            'starts_with_how_many': q_lower.startswith('how many'),
            'starts_with_where': q_lower.startswith('where'),
            'starts_with_is_are': q_lower.startswith(('is', 'are')),
            'starts_with_does_do': q_lower.startswith(('does', 'do')),
            'has_color_word': any(color in q_lower for color in self.ANSWER_CATEGORIES['colors']),
            'has_number_word': any(num in q_lower for num in ['many', 'number', 'count']),
            'has_spatial': any(word in q_lower for word in ['left', 'right', 'top', 'bottom', 'front', 'behind', 'next to', 'near']),
            'has_comparison': any(word in q_lower for word in ['more', 'less', 'bigger', 'smaller', 'taller', 'shorter']),
            'has_and': ' and ' in q_lower,
            'has_or': ' or ' in q_lower,
            'has_negation': any(word in q_lower for word in ['not', "n't", 'no ']),
            'asks_about_person': any(word in q_lower for word in ['person', 'people', 'man', 'woman', 'child', 'who'])
        }
        
        return features
    
    def check_success(self, action: Any) -> Tuple[bool, str]:
        """
        检查VQA答案
        
        支持yes/no答案和开放式答案
        """
        if action is None:
            return False, "No answer provided"
        
        # 规范化用户答案
        user_answer = self._normalize_answer(action)
        
        if user_answer is None or user_answer == "":
            return False, "Please provide a valid answer"
        
        # 获取正确答案
        correct_answer = self.expected_answer
        
        # 检查是否匹配
        if user_answer == correct_answer:
            consensus_info = ""
            if self.answer_distribution and len(self.answer_distribution) > 1:
                consensus = self._calculate_answer_consensus()
                if consensus < 0.5:
                    consensus_info = f" (Note: This question had diverse answers, consensus: {consensus:.1%})"
            return True, f"Correct! The answer is '{correct_answer}'{consensus_info}"
        else:
            # 对于开放式问题，检查是否是可接受的答案变体
            if not self.is_binary and self._is_acceptable_variant(user_answer, correct_answer):
                return True, f"Acceptable! (Answer: '{correct_answer}', you said: '{user_answer}')"
            
            # 提供有用的反馈
            if self.is_binary:
                explanation = self._get_error_explanation(user_answer, correct_answer)
                return False, f"Incorrect. The answer is '{correct_answer}'. {explanation}"
            else:
                # 对于开放式答案，提供更多上下文
                other_answers = ""
                if self.answer_distribution and len(self.answer_distribution) > 1:
                    top_answers = sorted(self.answer_distribution.items(), key=lambda x: x[1], reverse=True)[:3]
                    other_answers = f" Common answers: {', '.join([f'{ans} ({cnt})' for ans, cnt in top_answers])}"
                return False, f"Incorrect. The answer is '{correct_answer}' (you said '{user_answer}').{other_answers}"
    
    def _normalize_answer(self, answer: Any) -> Optional[str]:
        """规范化答案格式"""
        if answer is None:
            return None
        
        answer_str = str(answer).strip().lower()
        
        # 移除标点符号
        answer_str = re.sub(r'[.,!?;:\'"]+', '', answer_str)
        
        # 对于二元答案
        if self.is_binary:
            # 处理常见的yes变体
            if answer_str in ['yes', 'y', 'true', '1', 'correct', 'right', 'affirmative', 'yeah', 'yep', 'sure']:
                return 'yes'
            # 处理常见的no变体
            elif answer_str in ['no', 'n', 'false', '0', 'incorrect', 'wrong', 'negative', 'nope', 'nah']:
                return 'no'
            # 尝试从句子中提取yes/no
            elif re.search(r'\b(yes|yeah|yep)\b', answer_str) and not re.search(r'\b(no|not|nope)\b', answer_str):
                return 'yes'
            elif re.search(r'\b(no|nope|not)\b', answer_str) and not re.search(r'\b(yes|yeah)\b', answer_str):
                return 'no'
            else:
                return None
        else:
            # 对于开放式答案，进行基本清理
            # 移除冠词
            answer_str = re.sub(r'\b(a|an|the)\b', '', answer_str).strip()
            # 移除多余空格
            answer_str = ' '.join(answer_str.split())
            return answer_str
    
    def _is_acceptable_variant(self, user_answer: str, correct_answer: str) -> bool:
        """检查是否是可接受的答案变体"""
        # 完全匹配
        if user_answer == correct_answer:
            return True
        
        # 单复数变体
        if user_answer + 's' == correct_answer or user_answer == correct_answer + 's':
            return True
        
        # 同义词检查（简单示例）
        synonyms = {
            'person': ['people', 'man', 'woman', 'human'],
            'car': ['vehicle', 'automobile'],
            'street': ['road'],
            'building': ['house', 'structure'],
            # 可以扩展更多同义词
        }
        
        for key, values in synonyms.items():
            if correct_answer == key and user_answer in values:
                return True
            if user_answer == key and correct_answer in values:
                return True
        
        # 数字的不同表示
        number_words = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten'
        }
        for digit, word in number_words.items():
            if (user_answer == digit and correct_answer == word) or (user_answer == word and correct_answer == digit):
                return True
        
        return False
    
    def _get_error_explanation(self, user_answer: str, correct_answer: str) -> str:
        """生成错误解释（主要用于yes/no问题）"""
        explanations = []
        
        if self.task_type == 'visual_verification':
            if correct_answer == 'yes':
                explanations.append("The statement IS true based on the image.")
            else:
                explanations.append("The statement is NOT true based on the image.")
        
        if self.complexity_features.get('has_negation'):
            explanations.append("Note the negation in the question.")
        
        return " ".join(explanations) if explanations else ""
    
    def _calculate_answer_consensus(self) -> float:
        """计算答案一致性"""
        if not self.answer_distribution:
            return 1.0
        
        total_answers = sum(self.answer_distribution.values())
        if total_answers == 0:
            return 1.0
        
        # 最常见答案的比例
        max_count = max(self.answer_distribution.values())
        return max_count / total_answers
    
    def validate(
        self,
        chat_history: List[Dict],
        observation: Any,
        full_history: Optional[List[Any]] = None
    ) -> Tuple[float, bool, str, Dict[str, Any]]:
        """
        验证VQA任务执行情况
        """
        # 调用父类验证
        reward, done, message, info = super().validate(
            chat_history, observation, full_history
        )
        
        # 添加VQA特定的信息
        info["task_type"] = self.task_type
        info["question_type"] = self.question_type
        info["answer_type"] = self.answer_type
        info["expected_answer"] = self.expected_answer
        info["is_binary"] = self.is_binary
        info["is_complex"] = self.is_complex
        info["answer_length"] = self.answer_length_category
        info["answer_distribution"] = self.answer_distribution
        info["answer_consensus"] = self._calculate_answer_consensus()
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
            "user_answer": str(user_answer),
            "correct_answer": self.expected_answer,
            "normalized_user_answer": None,
            "is_common_answer": False
        }
        
        # 规范化用户答案
        normalized = self._normalize_answer(user_answer)
        error_info["normalized_user_answer"] = normalized
        
        if normalized is None or normalized == "":
            error_info["error_type"] = "invalid_format"
        else:
            # 检查是否是常见答案
            if self.answer_distribution and normalized in self.answer_distribution:
                error_info["is_common_answer"] = True
                error_info["answer_frequency"] = self.answer_distribution[normalized]
            
            # 分析错误类型
            if self.is_binary:
                error_info["error_type"] = "binary_confusion"
            elif self.task_type == 'object_counting':
                # 检查是否是数字错误
                try:
                    user_num = int(normalized) if normalized.isdigit() else None
                    correct_num = int(self.expected_answer) if self.expected_answer.isdigit() else None
                    if user_num is not None and correct_num is not None:
                        error_info["error_type"] = "counting_error"
                        error_info["count_difference"] = user_num - correct_num
                except:
                    error_info["error_type"] = "non_numeric_for_counting"
            elif self.task_type == 'color_recognition':
                if normalized in self.ANSWER_CATEGORIES['colors']:
                    error_info["error_type"] = "wrong_color"
                else:
                    error_info["error_type"] = "invalid_color"
            elif self.task_type == 'activity_recognition':
                if normalized in self.ANSWER_CATEGORIES['activities']:
                    error_info["error_type"] = "wrong_activity"
                else:
                    error_info["error_type"] = "activity_misinterpretation"
            else:
                # 检查答案长度差异
                user_length = len(normalized.split())
                expected_length = len(self.expected_answer.split())
                if abs(user_length - expected_length) > 2:
                    error_info["error_type"] = "length_mismatch"
                else:
                    error_info["error_type"] = "content_mismatch"
        
        return error_info
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取VQA特定的指标"""
        metrics = super().get_metrics()
        
        metrics.update({
            "task_type": self.task_type,
            "question_type": self.question_type,
            "answer_type": self.answer_type,
            "expected_answer": self.expected_answer,
            "is_binary": self.is_binary,
            "is_complex": self.is_complex,
            "answer_length": self.answer_length_category,
            "num_unique_answers": len(self.answer_distribution),
            "answer_consensus": self._calculate_answer_consensus(),
            "difficulty": self._assess_difficulty(),
            "question_features": self._analyze_question_features(),
            "entities": self.entities,
            "complexity_features": self.complexity_features
        })
        
        return metrics
    
    def _assess_difficulty(self) -> str:
        """评估任务难度"""
        difficulty_score = 0
        
        # 基于答案一致性（一致性低的更难）
        consensus = self._calculate_answer_consensus()
        if consensus < 0.3:
            difficulty_score += 3
        elif consensus < 0.5:
            difficulty_score += 2
        elif consensus < 0.7:
            difficulty_score += 1
        
        # 基于答案长度
        if self.answer_length_category == 'long':
            difficulty_score += 2
        elif self.answer_length_category == 'medium':
            difficulty_score += 1
        
        # 基于复杂度
        if self.is_complex:
            difficulty_score += 2
        
        # 基于任务类型
        difficult_tasks = ['visual_reasoning', 'temporal_reasoning', 'activity_recognition']
        if self.task_type in difficult_tasks:
            difficulty_score += 1
        
        # 基于是否开放式
        if not self.is_binary:
            difficulty_score += 1
        
        # 返回难度等级
        if difficulty_score >= 5:
            return "hard"
        elif difficulty_score >= 3:
            return "medium"
        else:
            return "easy"
    
    def get_observation(self) -> Dict[str, Any]:
        """获取任务观察"""
        obs = super().get_observation()
        
        # 添加VQA特定信息
        obs["scene_type"] = "COCO real-world image"
        obs["expected_content"] = "real objects, people, and scenes"
        obs["task_focus"] = "visual understanding and question answering"
        obs["answer_type"] = self.answer_type
        obs["answer_format"] = "binary (yes/no)" if self.is_binary else f"open-ended ({self.answer_length_category} answer)"
        
        if self.answer_distribution:
            obs["answer_diversity"] = len(self.answer_distribution)
            obs["answer_consensus"] = f"{self._calculate_answer_consensus():.1%}"
        
        return obs
    
    def get_task_description(self) -> str:
        """获取任务描述"""
        desc_parts = [
            f"This is a Visual Question Answering (VQA) task on a real-world COCO image.",
            f"Task type: {self.TASK_TYPES.get(self.task_type, self.task_type)}",
            f"Question type: {self.QUESTION_TYPES.get(self.question_type, self.question_type)}"
        ]
        
        if self.is_complex:
            desc_parts.append("⚠️ This is a complex question requiring careful analysis.")
        
        if self.is_binary:
            desc_parts.append(f"Expected answer: '{self.expected_answer}' (yes/no)")
        else:
            desc_parts.append(f"Expected answer: '{self.expected_answer}' ({self.answer_length_category} answer)")
        
        if self.answer_distribution and len(self.answer_distribution) > 1:
            consensus = self._calculate_answer_consensus()
            desc_parts.append(f"Answer consensus: {consensus:.1%}")
            if consensus < 0.5:
                desc_parts.append("Note: This question has diverse valid answers")
        
        desc_parts.append(f"Difficulty: {self._assess_difficulty()}")
        
        return "\n".join(desc_parts)