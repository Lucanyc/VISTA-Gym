#!/usr/bin/env python3
"""
A-OKVQA Task implementation for VLM Gym
Handles knowledge-based visual question answering with augmented OK-VQA dataset
"""

from typing import Tuple, Dict, Any, List, Optional
import re
from collections import Counter

from .vision_qa_task import VisionQATask


class AOKVQATask(VisionQATask):
    """
    A-OKVQA 特定任务
    
    专门处理需要外部知识的视觉问答任务，支持：
    - Multiple choice questions with 4 options
    - Open-ended questions with direct answers
    - Knowledge-based reasoning
    - Rationale-based explanations
    
    评分机制：
    - Multiple choice: exact match with correct choice
    - Open-ended: VQA accuracy = min(#humans that provided that answer / 3, 1)
    """
    
    # A-OKVQA的任务类型
    TASK_TYPES = {
        'vqa_knowledge': 'Questions requiring external knowledge',
        'vqa_occupation': 'Questions about occupations/jobs',
        'vqa_counting': 'Counting objects or quantity questions',
        'vqa_color': 'Questions about colors',
        'vqa_object': 'Object identification questions',
        'vqa_location': 'Questions about position or location',
        'vqa_activity': 'Questions about actions or activities',
        'vqa_binary': 'Yes/No questions',
        'vqa_general': 'General knowledge questions'
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
        # Common synonyms
        'taxi': 'cab', 'cab': 'taxi',  # For transportation
        'physician': 'doctor', 'doc': 'doctor'
    }
    
    @classmethod
    def get_task_id(cls) -> str:
        """获取任务类型ID"""
        return "vlm-gym.aokvqa"
    
    def setup(self) -> Tuple[str, Dict[str, Any]]:
        """设置A-OKVQA特定的任务"""
        # 调用父类setup
        task_goal, task_info = super().setup()
        
        # 从metadata中提取A-OKVQA特定信息
        self.question_id = self.metadata.get('question_id', '')
        self.image_index = self.metadata.get('image_index', -1)
        self.image_name = self.metadata.get('image_name', '')
        self.task_type = task_info.get('task', 'vqa_knowledge')
        
        # A-OKVQA特有的信息
        self.choices = self.task_data.get('choices', None)
        self.correct_choice_idx = self.metadata.get('correct_choice_idx', -1)
        self.direct_answers = self.metadata.get('direct_answers', [])
        self.rationales = self.metadata.get('rationales', [])
        self.difficult_direct_answer = self.metadata.get('difficult_direct_answer', False)
        
        # 判断是否是多选题模式
        self.is_multiple_choice = self.choices is not None
        
        # 分析问题特征
        self.requires_knowledge = self.task_type == 'vqa_knowledge'
        self.is_occupation = self.task_type == 'vqa_occupation'
        self.is_difficult = self.difficult_direct_answer
        
        # 计算直接答案的一致性
        self.direct_answer_agreement = self._calculate_direct_answer_agreement()
        
        # 添加A-OKVQA特定信息到task_info
        task_info["question_id"] = self.question_id
        task_info["image_index"] = self.image_index
        task_info["image_name"] = self.image_name
        task_info["task_type"] = self.task_type
        task_info["is_multiple_choice"] = self.is_multiple_choice
        task_info["is_difficult"] = self.is_difficult
        task_info["requires_knowledge"] = self.requires_knowledge
        task_info["direct_answer_agreement"] = self.direct_answer_agreement
        task_info["has_rationales"] = len(self.rationales) > 0
        task_info["dataset"] = "aokvqa"
        
        # 修改任务目标以包含A-OKVQA特定指导
        enhanced_goal = self._enhance_task_goal(task_goal)
        
        return enhanced_goal, task_info
    
    def _enhance_task_goal(self, base_goal: str) -> str:
        """增强任务目标描述，添加A-OKVQA特定的指导"""
        enhanced_parts = [base_goal]
        
        # 如果是多选题
        if self.is_multiple_choice:
            enhanced_parts.append("\n**This is a multiple choice question.**")
            enhanced_parts.append("Choose the most appropriate answer from the given options:")
            for i, choice in enumerate(self.choices):
                enhanced_parts.append(f"  {i}: {choice}")
            enhanced_parts.append("Provide your answer as the option number (0-3) or the text of your chosen option.")
        
        # 根据任务类型添加特定指导
        if self.requires_knowledge:
            enhanced_parts.append("\n**This question requires external knowledge.**")
            enhanced_parts.append("- Use both visual information and general knowledge to answer")
            enhanced_parts.append("- Consider common sense and real-world facts")
            enhanced_parts.append("- The answer may not be directly visible in the image")
            
        elif self.is_occupation:
            enhanced_parts.append("\n**This is an occupation/job-related question.**")
            enhanced_parts.append("- Look for visual cues about the person's profession")
            enhanced_parts.append("- Consider clothing, tools, environment, and activities")
            enhanced_parts.append("- Use context clues to infer the occupation")
        
        # 如果问题被标记为困难
        if self.is_difficult:
            enhanced_parts.append("\n**Note**: This question has been marked as difficult with low human agreement.")
            enhanced_parts.append("Multiple interpretations may be valid. Provide your best reasoned answer.")
        
        # 如果有推理解释
        if self.rationales:
            enhanced_parts.append("\n**Reasoning hints:**")
            enhanced_parts.append("Consider the following type of reasoning:")
            # 只显示第一个rationale作为提示
            hint = self.rationales[0][:100] + "..." if len(self.rationales[0]) > 100 else self.rationales[0]
            enhanced_parts.append(f"- {hint}")
        
        # 添加通用的A-OKVQA提示
        enhanced_parts.append("\n**A-OKVQA Tips:**")
        enhanced_parts.append("- Combine visual observation with world knowledge")
        enhanced_parts.append("- Think about what information is needed beyond what's visible")
        enhanced_parts.append("- Consider the context and make reasonable inferences")
        
        return "\n".join(enhanced_parts)
    
    def _calculate_direct_answer_agreement(self) -> float:
        """计算直接答案的一致性程度"""
        if not self.direct_answers:
            return 0.0
        
        # 统计答案频率
        answer_counts = Counter(self.direct_answers)
        total = len(self.direct_answers)
        
        if total == 0:
            return 0.0
        
        # 最常见答案的比例
        max_count = max(answer_counts.values())
        return max_count / total
    
    def check_success(self, action: Any) -> Tuple[bool, str]:
        """
        检查A-OKVQA答案
        
        支持两种模式：
        1. 多选题模式：精确匹配正确选项
        2. 开放式模式：使用VQA评分机制
        """
        if not action:
            return False, "No answer provided"
        
        user_answer = str(action).strip()
        
        # 多选题模式
        if self.is_multiple_choice:
            # 检查是否是数字索引
            try:
                choice_idx = int(user_answer)
                if 0 <= choice_idx < len(self.choices):
                    if choice_idx == self.correct_choice_idx:
                        return True, f"Correct! The answer is {self.choices[choice_idx]}"
                    else:
                        correct_answer = self.choices[self.correct_choice_idx]
                        return False, f"Incorrect. The correct answer is {self.correct_choice_idx}: {correct_answer}"
            except ValueError:
                pass
            
            # 检查是否匹配选项文本
            normalized_user = self._normalize_answer(user_answer)
            for idx, choice in enumerate(self.choices):
                if self._normalize_answer(choice) == normalized_user:
                    if idx == self.correct_choice_idx:
                        return True, f"Correct! The answer is {choice}"
                    else:
                        correct_answer = self.choices[self.correct_choice_idx]
                        return False, f"Incorrect. The correct answer is {correct_answer}"
            
            # 没有匹配到任何选项
            correct_answer = self.choices[self.correct_choice_idx]
            return False, f"Please choose from the given options. The correct answer is {self.correct_choice_idx}: {correct_answer}"
        
        # 开放式模式 - 使用直接答案评分
        normalized_user = self._normalize_answer(user_answer)
        
        if self.direct_answers:
            # 统计匹配的直接答案数量
            match_count = sum(1 for da in self.direct_answers 
                            if self._normalize_answer(da) == normalized_user)
            
            # 计算VQA分数
            vqa_score = min(match_count / 3.0, 1.0)
            
            if vqa_score >= 1.0:
                return True, f"Correct! {match_count}/{len(self.direct_answers)} human annotators gave the same answer"
            elif vqa_score >= 0.67:
                return True, f"Acceptable answer! {match_count}/{len(self.direct_answers)} annotators agreed"
            elif match_count > 0:
                # 显示最常见的答案
                answer_counts = Counter(self.direct_answers)
                top_answer = answer_counts.most_common(1)[0][0]
                return False, f"Partially correct ({match_count} annotators). Most common answer: {top_answer}"
            else:
                # 显示前3个最常见的答案
                answer_counts = Counter(self.direct_answers)
                top_answers = [ans for ans, _ in answer_counts.most_common(3)]
                return False, f"Incorrect. Common answers: {', '.join(top_answers)}"
        else:
            # 没有直接答案信息，使用简单匹配
            correct_normalized = self._normalize_answer(str(self.answer))
            if normalized_user == correct_normalized:
                return True, f"Correct! The answer is {self.answer}"
            else:
                return False, f"Incorrect. The expected answer is {self.answer}"
    
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
        验证A-OKVQA任务执行情况
        """
        # 调用父类验证
        reward, done, message, info = super().validate(
            chat_history, observation, full_history
        )
        
        # 添加A-OKVQA特定的信息
        info["question_id"] = self.question_id
        info["image_index"] = self.image_index
        info["task_type"] = self.task_type
        info["is_multiple_choice"] = self.is_multiple_choice
        info["is_difficult"] = self.is_difficult
        info["requires_knowledge"] = self.requires_knowledge
        info["direct_answer_agreement"] = self.direct_answer_agreement
        info["has_rationales"] = len(self.rationales) > 0
        
        # 如果提供了答案，进行详细分析
        if info.get("answer_provided"):
            provided_answer = str(info["answer_provided"])
            normalized_answer = self._normalize_answer(provided_answer)
            
            if self.is_multiple_choice:
                # 检查是否选择了正确选项
                try:
                    choice_idx = int(provided_answer)
                    info["selected_choice_idx"] = choice_idx
                    info["selected_correct_choice"] = (choice_idx == self.correct_choice_idx)
                except:
                    # 尝试匹配文本
                    for idx, choice in enumerate(self.choices):
                        if self._normalize_answer(choice) == normalized_answer:
                            info["selected_choice_idx"] = idx
                            info["selected_correct_choice"] = (idx == self.correct_choice_idx)
                            break
            
            # 计算直接答案匹配
            if self.direct_answers:
                match_count = sum(1 for da in self.direct_answers 
                                if self._normalize_answer(da) == normalized_answer)
                info["direct_answer_matches"] = match_count
                info["direct_answer_score"] = min(match_count / 3.0, 1.0)
        
        return reward, done, message, info
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取A-OKVQA特定的指标"""
        metrics = super().get_metrics()
        
        metrics.update({
            "question_id": self.question_id,
            "image_index": self.image_index,
            "image_name": self.image_name,
            "task_type": self.task_type,
            "is_multiple_choice": self.is_multiple_choice,
            "num_choices": len(self.choices) if self.choices else 0,
            "is_difficult": self.is_difficult,
            "requires_knowledge": self.requires_knowledge,
            "direct_answer_agreement": self.direct_answer_agreement,
            "num_direct_answers": len(self.direct_answers),
            "num_unique_direct_answers": len(set(self.direct_answers)),
            "num_rationales": len(self.rationales),
            "difficulty_factors": self.get_difficulty_factors()
        })
        
        return metrics
    
    def get_difficulty_factors(self) -> List[str]:
        """获取影响难度的因素"""
        factors = []
        
        # 基于标记
        if self.is_difficult:
            factors.append("marked_difficult")
        
        # 基于直接答案一致性
        if self.direct_answer_agreement < 0.3:
            factors.append("high_disagreement")
        elif self.direct_answer_agreement < 0.5:
            factors.append("moderate_disagreement")
        
        # 基于答案多样性
        if len(set(self.direct_answers)) > 5:
            factors.append("diverse_answers")
        
        # 基于任务类型
        if self.requires_knowledge:
            factors.append("requires_external_knowledge")
        
        if self.is_occupation:
            factors.append("occupation_inference")
        
        # 基于问题长度
        if self.question and len(self.question.split()) > 15:
            factors.append("complex_question")
        
        return factors
    
    def get_hint(self) -> str:
        """获取问题提示（用于辅助）"""
        hints = []
        
        if self.is_multiple_choice:
            hints.append("Choose from the given options based on the image and your knowledge.")
        
        if self.requires_knowledge:
            hints.append("This question requires knowledge beyond what's visible in the image.")
        elif self.is_occupation:
            hints.append("Look for clues about the person's job or profession.")
        
        # 如果有低一致性，提醒
        if self.direct_answer_agreement < 0.5:
            hints.append("Note: This question has multiple valid interpretations.")
        
        # 提供一个简化的rationale提示
        if self.rationales and len(self.rationales) > 0:
            # 提取关键词作为提示
            first_rationale = self.rationales[0].lower()
            if "because" in first_rationale:
                reason_part = first_rationale.split("because")[1].strip()
                hints.append(f"Consider: {reason_part[:50]}...")
        
        return " ".join(hints) if hints else "Analyze the image and use your knowledge to answer the question."