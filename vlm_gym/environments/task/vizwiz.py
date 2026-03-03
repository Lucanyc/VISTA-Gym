#!/usr/bin/env python3
"""
VizWiz Task implementation for VLM Gym
Handles visual question answering designed for visually impaired users
"""

from typing import Tuple, Dict, Any, List, Optional
import re
from collections import Counter
import difflib

from .vision_qa_task import VisionQATask


class VizWizTask(VisionQATask):
    """
    VizWiz 特定任务
    
    专门处理为视觉障碍用户设计的视觉问答任务，特点包括：
    - 真实场景的图片（可能模糊、角度不佳）
    - 日常生活相关的问题
    - 多个标注者的答案（带置信度）
    - 部分问题可能无法回答
    
    支持的答案类型：
    - 开放式文本答案
    - 颜色识别
    - 物体识别
    - 文字阅读（OCR相关）
    - 数量计数
    - Yes/No判断
    """
    
    # VizWiz常见的答案类型分类
    ANSWER_CATEGORIES = {
        'color': 'Color identification questions',
        'object': 'Object recognition questions',
        'text_reading': 'Text/label reading questions',
        'counting': 'Counting objects questions',
        'description': 'Scene description questions',
        'yes_no': 'Binary yes/no questions'
    }
    
    # 常见的颜色词汇
    COMMON_COLORS = {
        'red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 
        'purple', 'orange', 'pink', 'gray', 'grey', 'silver', 'gold',
        'beige', 'tan', 'navy', 'maroon', 'teal', 'turquoise'
    }
    
    # 常见的无法回答原因
    UNANSWERABLE_REASONS = {
        'blurry': 'Image is too blurry',
        'dark': 'Image is too dark',
        'cutoff': 'Object is cut off or partially visible',
        'unclear': 'Question or image is unclear',
        'no_text': 'No text visible in image'
    }
    
    @classmethod
    def get_task_id(cls) -> str:
        """获取任务类型ID"""
        return "vlm-gym.vizwiz"
    
    def setup(self) -> Tuple[str, Dict[str, Any]]:
        """设置VizWiz特定的任务"""
        # 调用父类setup
        task_goal, task_info = super().setup()
        
        # 从metadata中提取VizWiz特定信息
        self.answer_type = self.metadata.get('answer_type', 'other')
        self.answerable = self.metadata.get('answerable', 1)
        self.answer_confidence = self.metadata.get('answer_confidence', 0.0)
        self.all_answers = self.metadata.get('all_answers', [])
        self.answer_count = self.metadata.get('answer_count', 10)
        self.confidence_distribution = self.metadata.get('confidence_distribution', {})
        
        # 分析问题类型
        self.question_category = self._categorize_question()
        self.is_unanswerable = (self.answerable == 0 or self.answer.lower() == 'unanswerable')
        self.is_high_confidence = self.answer_confidence >= 0.7
        self.is_unanimous = self.answer_confidence >= 0.9
        
        # 检测特定问题类型
        self.is_color_question = self._is_color_question()
        self.is_text_reading = self._is_text_reading_question()
        self.is_counting = self._is_counting_question()
        self.is_yes_no = self._is_yes_no_question()
        
        # 添加VizWiz特定的处理
        task_info["answer_type"] = self.answer_type
        task_info["answerable"] = self.answerable
        task_info["answer_confidence"] = self.answer_confidence
        task_info["is_unanswerable"] = self.is_unanswerable
        task_info["is_high_confidence"] = self.is_high_confidence
        task_info["question_category"] = self.question_category
        task_info["all_possible_answers"] = self.all_answers
        task_info["annotator_agreement"] = self._calculate_annotator_agreement()
        task_info["dataset"] = "vizwiz"
        
        # 修改任务目标以包含VizWiz特定指导
        enhanced_goal = self._enhance_task_goal(task_goal)
        
        return enhanced_goal, task_info
    
    def _enhance_task_goal(self, base_goal: str) -> str:
        """增强任务目标描述，添加VizWiz特定的指导"""
        enhanced_parts = [base_goal]
        
        # 添加VizWiz数据集的特殊说明
        enhanced_parts.append("\n**Note: This is from the VizWiz dataset, designed for assisting visually impaired users.**")
        enhanced_parts.append("The image may have quality issues (blur, poor lighting, unusual angles).")
        
        # 如果是低置信度答案，提醒
        if self.answer_confidence < 0.5:
            enhanced_parts.append(f"\nThis question has low annotator agreement (confidence: {self.answer_confidence:.2f}).")
            enhanced_parts.append("Multiple valid answers may exist.")
        
        # 根据问题类型添加指导
        if self.is_color_question:
            enhanced_parts.append("\nThis is a color identification question. Focus on the dominant colors visible.")
        elif self.is_text_reading:
            enhanced_parts.append("\nThis is a text reading question. Try to identify any visible text, labels, or writing.")
        elif self.is_counting:
            enhanced_parts.append("\nThis is a counting question. Count the specified objects carefully.")
        elif self.is_yes_no:
            enhanced_parts.append("\nThis is a yes/no question. Provide a clear 'Yes' or 'No' answer.")
        
        # 如果可能无法回答
        if self.is_unanswerable:
            enhanced_parts.append("\n**Important**: This image might be unanswerable due to quality issues.")
            enhanced_parts.append("If you cannot determine the answer due to image quality, say 'unanswerable'.")
        
        # 添加通用提示
        enhanced_parts.append("\nWhen analyzing the image:")
        enhanced_parts.append("- Be helpful and specific in your answer")
        enhanced_parts.append("- If the image quality is poor, mention what you can still observe")
        enhanced_parts.append("- Provide the most useful information possible for a visually impaired user")
        
        # 如果有多个可能的答案
        if len(self.all_answers) > 3:
            enhanced_parts.append(f"\nNote: Annotators provided {len(self.all_answers)} different answers for this question.")
        
        return "\n".join(enhanced_parts)
    
    def _categorize_question(self) -> str:
        """分类问题类型"""
        if not self.question:
            return 'unknown'
        
        question_lower = self.question.lower()
        
        # 颜色相关
        if any(word in question_lower for word in ['color', 'colour', 'what color']):
            return 'color'
        
        # 文字阅读
        if any(word in question_lower for word in ['read', 'say', 'text', 'label', 'write', 'written']):
            return 'text_reading'
        
        # 计数
        if any(word in question_lower for word in ['how many', 'count', 'number of']):
            return 'counting'
        
        # 物体识别
        if any(word in question_lower for word in ['what is this', 'what kind', 'what type', 'identify']):
            return 'object'
        
        # Yes/No
        if any(word in question_lower for word in ['is this', 'is it', 'are there', 'can you', 'does']):
            return 'yes_no'
        
        # 描述
        if any(word in question_lower for word in ['describe', 'what do you see', 'tell me']):
            return 'description'
        
        return 'other'
    
    def _is_color_question(self) -> bool:
        """检查是否是颜色相关问题"""
        return self.question_category == 'color' or \
               any(color in self.question.lower() for color in ['color', 'colour'])
    
    def _is_text_reading_question(self) -> bool:
        """检查是否是文字阅读问题"""
        return self.question_category == 'text_reading' or \
               any(word in self.question.lower() for word in ['read', 'say', 'text', 'label'])
    
    def _is_counting_question(self) -> bool:
        """检查是否是计数问题"""
        return self.question_category == 'counting' or 'how many' in self.question.lower()
    
    def _is_yes_no_question(self) -> bool:
        """检查是否是Yes/No问题"""
        # 检查答案
        if self.answer.lower() in ['yes', 'no']:
            return True
        # 检查问题模式
        question_lower = self.question.lower()
        yes_no_patterns = ['is this', 'is it', 'is there', 'are there', 'does', 'do', 'can', 'will', 'would']
        return any(pattern in question_lower for pattern in yes_no_patterns)
    
    def _calculate_annotator_agreement(self) -> float:
        """计算标注者一致性"""
        if not self.all_answers:
            return 0.0
        
        # 计算最常见答案的比例
        answer_counts = Counter(ans.lower() for ans in self.all_answers)
        if answer_counts:
            most_common_count = answer_counts.most_common(1)[0][1]
            return most_common_count / len(self.all_answers)
        return 0.0
    
    def check_success(self, action: Any) -> Tuple[bool, str]:
        """
        检查VizWiz答案
        
        由于VizWiz答案的多样性，采用更灵活的匹配策略
        """
        if not action:
            return False, "No answer provided"
        
        # 清理答案格式
        user_answer = str(action).strip().lower()
        correct_answer = str(self.answer).strip().lower()
        
        # 1. 处理unanswerable情况
        if self.is_unanswerable:
            unanswerable_words = ['unanswerable', 'cannot answer', 'can\'t answer', 'unable to answer', 
                                'not visible', 'too blurry', 'too dark', 'unclear']
            if any(word in user_answer for word in unanswerable_words):
                return True, "Correct! The image is indeed unanswerable."
            # 如果用户提供了具体答案但图片标记为unanswerable，给予部分认可
            if len(user_answer) > 10:  # 用户尝试回答
                return False, "The image was marked as unanswerable by annotators, but your attempt to help is appreciated."
        
        # 2. Yes/No答案
        if self.is_yes_no:
            user_yes_no = self._extract_yes_no(user_answer)
            if user_yes_no and user_yes_no == correct_answer:
                return True, f"Correct! The answer is {correct_answer}."
            elif user_yes_no:
                return False, f"Incorrect. The answer is {correct_answer}."
        
        # 3. 精确匹配
        if user_answer == correct_answer:
            return True, f"Correct! The answer is '{self.answer}'."
        
        # 4. 包含匹配（用户答案包含正确答案）
        if correct_answer in user_answer:
            return True, f"Correct! You identified '{self.answer}'."
        
        # 5. 检查是否匹配任何标注者的答案
        if self.all_answers:
            for alt_answer in self.all_answers:
                alt_lower = alt_answer.lower()
                if alt_lower in user_answer or user_answer in alt_lower:
                    confidence_msg = f" (confidence: {self.answer_confidence:.2f})" if self.answer_confidence < 0.7 else ""
                    return True, f"Acceptable answer! Your answer matches one of the annotator responses{confidence_msg}."
        
        # 6. 相似度匹配（对于较长的答案）
        if len(correct_answer) > 3 and len(user_answer) > 3:
            similarity = difflib.SequenceMatcher(None, user_answer, correct_answer).ratio()
            if similarity > 0.8:
                return True, f"Close enough! The exact answer was '{self.answer}'."
            elif similarity > 0.6:
                return False, f"Partial match. The correct answer is '{self.answer}'."
        
        # 7. 颜色问题的特殊处理
        if self.is_color_question:
            user_colors = self._extract_colors(user_answer)
            correct_colors = self._extract_colors(correct_answer)
            if user_colors and correct_colors and user_colors & correct_colors:
                return True, f"Correct! You identified the color(s): {', '.join(user_colors & correct_colors)}."
        
        # 8. 数字答案的特殊处理
        if self.is_counting:
            user_num = self._extract_number(user_answer)
            correct_num = self._extract_number(correct_answer)
            if user_num is not None and correct_num is not None:
                if user_num == correct_num:
                    return True, f"Correct! The count is {int(correct_num)}."
                else:
                    return False, f"Incorrect count. The correct answer is {int(correct_num)}."
        
        # 9. 低置信度答案的宽松处理
        if self.answer_confidence < 0.5 and len(user_answer) > 5:
            # 对于低置信度答案，如果用户提供了合理的描述，认为部分正确
            return False, f"Your answer differs from the annotators' consensus ('{self.answer}'), but this question had low agreement among annotators."
        
        return False, f"Incorrect. The correct answer is '{self.answer}'."
    
    def _extract_yes_no(self, text: str) -> Optional[str]:
        """从文本中提取Yes/No答案"""
        text_lower = text.lower().strip()
        
        # 直接匹配
        if text_lower in ['yes', 'no']:
            return text_lower
        
        # 开头匹配
        if text_lower.startswith('yes'):
            return 'yes'
        elif text_lower.startswith('no'):
            return 'no'
        
        # 包含匹配（避免歧义）
        if 'yes' in text_lower and 'no' not in text_lower:
            return 'yes'
        elif 'no' in text_lower and 'yes' not in text_lower:
            return 'no'
        
        return None
    
    def _extract_colors(self, text: str) -> set:
        """从文本中提取颜色词汇"""
        text_lower = text.lower()
        found_colors = set()
        
        for color in self.COMMON_COLORS:
            if color in text_lower:
                found_colors.add(color)
        
        return found_colors
    
    def _extract_number(self, text: str) -> Optional[float]:
        """从文本中提取数字"""
        # 查找所有数字
        numbers = re.findall(r'\d+', text)
        if numbers:
            # 返回第一个数字
            return float(numbers[0])
        
        # 处理文字形式的数字
        word_to_num = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
            'ten': 10, 'eleven': 11, 'twelve': 12
        }
        
        text_lower = text.lower()
        for word, num in word_to_num.items():
            if word in text_lower:
                return float(num)
        
        return None
    
    def validate(
        self,
        chat_history: List[Dict],
        observation: Any,
        full_history: Optional[List[Any]] = None
    ) -> Tuple[float, bool, str, Dict[str, Any]]:
        """
        验证VizWiz任务执行情况
        """
        # 调用父类验证
        reward, done, message, info = super().validate(
            chat_history, observation, full_history
        )
        
        # 添加VizWiz特定的信息
        info["answer_type"] = self.answer_type
        info["answerable"] = self.answerable
        info["answer_confidence"] = self.answer_confidence
        info["is_unanswerable"] = self.is_unanswerable
        info["question_category"] = self.question_category
        info["annotator_agreement"] = self._calculate_annotator_agreement()
        info["all_possible_answers"] = self.all_answers
        
        # 分析答案质量
        if info.get("answer_provided"):
            provided_answer = str(info["answer_provided"]).lower()
            
            # 检查是否识别出unanswerable
            if self.is_unanswerable:
                unanswerable_detected = any(word in provided_answer for word in ['unanswerable', 'cannot', 'unclear'])
                info["correctly_identified_unanswerable"] = unanswerable_detected
            
            # 检查答案是否在所有可能答案中
            if self.all_answers:
                matches_any = any(ans.lower() in provided_answer or provided_answer in ans.lower() 
                                for ans in self.all_answers)
                info["matches_any_annotator"] = matches_any
            
            # 答案详细程度
            info["answer_length"] = len(provided_answer)
            info["is_detailed_answer"] = len(provided_answer) > 50
            
            # 对于低置信度问题的特殊处理
            if self.answer_confidence < 0.5:
                info["low_confidence_question"] = True
                # 放宽成功标准
                if reward == 0 and len(provided_answer) > 10:
                    # 给予部分奖励
                    reward = 0.5
                    message += " (Partial credit for attempting a low-confidence question)"
        
        return reward, done, message, info
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取VizWiz特定的指标"""
        metrics = super().get_metrics()
        
        metrics.update({
            "answer_type": self.answer_type,
            "answerable": self.answerable,
            "answer_confidence": self.answer_confidence,
            "is_unanswerable": self.is_unanswerable,
            "is_high_confidence": self.is_high_confidence,
            "is_unanimous": self.is_unanimous,
            "question_category": self.question_category,
            "annotator_count": self.answer_count,
            "unique_answers": len(self.all_answers) if self.all_answers else 0,
            "confidence_distribution": self.confidence_distribution,
            "is_color_question": self.is_color_question,
            "is_text_reading": self.is_text_reading,
            "is_counting": self.is_counting,
            "is_yes_no": self.is_yes_no,
            "difficulty_factors": self.get_difficulty_factors()
        })
        
        return metrics
    
    def get_difficulty_factors(self) -> List[str]:
        """获取影响难度的因素"""
        factors = []
        
        # 基于置信度
        if self.answer_confidence < 0.3:
            factors.append("very_low_agreement")
        elif self.answer_confidence < 0.5:
            factors.append("low_agreement")
        
        # 基于可回答性
        if self.is_unanswerable:
            factors.append("potentially_unanswerable")
        
        # 基于答案多样性
        if len(self.all_answers) > 5:
            factors.append("high_answer_diversity")
        
        # 基于问题类型
        if self.is_text_reading:
            factors.append("requires_ocr")
        if self.is_counting:
            factors.append("requires_counting")
        
        # 基于问题长度
        if self.question and len(self.question) > 50:
            factors.append("complex_question")
        
        return factors
    
    def get_assistance_hints(self) -> List[str]:
        """获取辅助提示（用于帮助视觉障碍用户）"""
        hints = []
        
        if self.is_color_question:
            hints.append("Focus on describing the main colors visible in the image.")
        
        if self.is_text_reading:
            hints.append("Try to read any text, labels, or signs visible in the image.")
        
        if self.is_counting:
            hints.append("Count the objects carefully and provide the exact number.")
        
        if self.answer_confidence < 0.5:
            hints.append("This question has multiple valid interpretations. Provide your best assessment.")
        
        if self.is_unanswerable:
            hints.append("If the image quality prevents answering, explain what you can observe.")
        
        return hints
    
    def format_for_accessibility(self, answer: str) -> str:
        """格式化答案以提高可访问性"""
        # 为视觉障碍用户优化答案格式
        formatted = answer
        
        # 确保颜色描述清晰
        if self.is_color_question:
            # 添加颜色描述的上下文
            if not any(word in answer.lower() for word in ['color is', 'colors are']):
                formatted = f"The color is {answer}" if len(answer.split()) <= 2 else answer
        
        # 确保计数答案清晰
        if self.is_counting:
            number = self._extract_number(answer)
            if number is not None:
                formatted = f"There are {int(number)} items" if "there" not in answer.lower() else answer
        
        # 确保yes/no答案清晰
        if self.is_yes_no and answer.lower() in ['yes', 'no']:
            formatted = f"{answer.capitalize()}."
        
        return formatted