"""
Text-VQA Task implementation for VLM Gym
Handles text-based visual question answering in natural images
"""

from typing import Tuple, Dict, Any, List, Optional
import re
import json
from pathlib import Path
from collections import Counter

from .vision_qa_task import VisionQATask


class TextVQATask(VisionQATask):
    """
    Text-VQA 特定任务
    
    专门处理场景文本相关的视觉问答任务，包括：
    - 文本阅读（读取图像中的文字）
    - 品牌识别（识别标志和品牌名称）
    - 时间信息（读取时钟、日期等）
    - 数字识别（价格、电话号码、车牌等）
    - 标志理解（交通标志、店铺招牌等）
    
    支持需要OCR能力的自然图像理解任务
    """
    
    # 文本问题类型
    QUESTION_TYPES = {
        'text_reading': '文本阅读（直接读取文字）',
        'brand_recognition': '品牌识别（标志、商标）',
        'temporal': '时间信息（时钟、日期）',
        'numeric': '数字信息（价格、号码）',
        'location': '位置信息（地址、方向）',
        'color': '颜色相关（文字颜色）',
        'counting': '计数问题（文字数量）',
        'yes_no': '是非判断',
        'what_which': '什么/哪个类问题',
        'other': '其他类型'
    }
    
    # 任务类型映射
    TASK_TYPES = {
        'text_reading': '纯文本阅读',
        'text_brand_recognition': '品牌/标志识别',
        'text_temporal_qa': '时间相关问答',
        'text_price_qa': '价格相关问答',
        'text_location_qa': '位置相关问答',
        'text_number_qa': '数字相关问答',
        'text_vqa': '通用文本问答'
    }
    
    def __init__(self, task_id: str, adapter: Any):
        """
        初始化Text-VQA任务
        
        Args:
            task_id: 任务ID
            adapter: Text-VQA数据适配器
        """
        # 调用父类初始化
        super().__init__(task_id, adapter)
        
        # 初始化Text-VQA特定属性
        self.task_type = None
        self.question_type = None
        self.has_ocr = False
        self.ocr_tokens = []
        self.all_answers = []
        self.answer_frequencies = {}
        self.requires_text_reading = True
        self.is_numeric = False
        self.is_multi_answer = False
        
        # 获取任务数据
        task_data = adapter.get_task_data(task_id)
        self.dataset_name = task_data.get('dataset', 'textvqa')
        self.task_type = task_data.get('task', 'text_vqa')
    
    @classmethod
    def get_task_id(cls) -> str:
        """获取任务类型ID"""
        return "vlm-gym.textvqa"
    
    def setup(self) -> Tuple[str, Dict[str, Any]]:
        """设置Text-VQA特定的任务"""
        # 调用父类setup
        task_goal, task_info = super().setup()
        
        # 从metadata中提取Text-VQA特定信息
        self.all_answers = self.metadata.get('all_answers', [self.answer] if self.answer else [])
        self.answer_frequencies = self.metadata.get('answer_frequency', {})
        self.ocr_tokens = self.metadata.get('ocr_tokens', [])
        self.has_ocr = len(self.ocr_tokens) > 0
        
        # 分析问题和答案特征
        self.question_type = self._classify_question_type()
        self.is_numeric = self._is_numeric_answer(self.answer)
        self.is_multi_answer = len(set(ans.lower() for ans in self.all_answers)) > 1
        
        # 检查答案是否在OCR tokens中
        answer_in_ocr = self._check_answer_in_ocr()
        
        # 添加Text-VQA特定的信息
        task_info["task_type"] = self.task_type
        task_info["question_type"] = self.question_type
        task_info["has_ocr"] = self.has_ocr
        task_info["ocr_token_count"] = len(self.ocr_tokens)
        task_info["is_numeric"] = self.is_numeric
        task_info["is_multi_answer"] = self.is_multi_answer
        task_info["answer_in_ocr"] = answer_in_ocr
        task_info["unique_answer_count"] = len(set(self.all_answers))
        task_info["dataset"] = "textvqa"
        task_info["text_complexity"] = self._assess_text_complexity()
        task_info["requires_text_reading"] = self.requires_text_reading
        
        # 修改任务目标以包含Text-VQA特定指导
        enhanced_goal = self._enhance_task_goal(task_goal)
        
        return enhanced_goal, task_info
    
    def _enhance_task_goal(self, base_goal: str) -> str:
        """增强任务目标描述，添加Text-VQA特定的指导"""
        enhanced_parts = [base_goal]
        
        # 添加文本阅读通用指导
        enhanced_parts.append("\n**Text Reading Guidelines:**")
        enhanced_parts.append("1. Carefully examine all text visible in the image")
        enhanced_parts.append("2. Look for text on signs, products, displays, etc.")
        enhanced_parts.append("3. Pay attention to text clarity and orientation")
        enhanced_parts.append("4. Read text exactly as it appears")
        
        # 根据任务类型添加特定指导
        if self.task_type == 'text_reading' or self.question_type == 'text_reading':
            enhanced_parts.append("\n**Direct Text Reading:**")
            enhanced_parts.append("- Focus on the specific text mentioned in the question")
            enhanced_parts.append("- Read complete words/phrases without interpretation")
            enhanced_parts.append("- Be precise with spelling and capitalization")
            
        elif self.task_type == 'text_brand_recognition' or self.question_type == 'brand_recognition':
            enhanced_parts.append("\n**Brand Recognition Tips:**")
            enhanced_parts.append("- Look for logos, brand names, company signs")
            enhanced_parts.append("- Check product labels and packaging")
            enhanced_parts.append("- Identify both text and visual brand elements")
            
        elif self.task_type == 'text_temporal_qa' or self.question_type == 'temporal':
            enhanced_parts.append("\n**Time/Date Reading Tips:**")
            enhanced_parts.append("- Look for clocks, calendars, date displays")
            enhanced_parts.append("- Read time in the format shown (digital/analog)")
            enhanced_parts.append("- Include AM/PM if visible")
            
        elif self.task_type == 'text_number_qa' or self.question_type == 'numeric':
            enhanced_parts.append("\n**Number Reading Tips:**")
            enhanced_parts.append("- Look for prices, phone numbers, addresses")
            enhanced_parts.append("- Include currency symbols if present")
            enhanced_parts.append("- Read complete number sequences")
        
        # 根据问题类型添加指导
        if self.question_type == 'location':
            enhanced_parts.append("\n**Location Text Tips:**")
            enhanced_parts.append("- Look for street signs, addresses, directions")
            enhanced_parts.append("- Check building names and landmarks")
            
        elif self.question_type == 'counting':
            enhanced_parts.append("\n**Counting Tips:**")
            enhanced_parts.append("- Count all instances of the requested text/item")
            enhanced_parts.append("- Check all visible areas of the image")
        
        # OCR相关指导
        if self.has_ocr:
            enhanced_parts.append(f"\n**OCR Information Available:**")
            enhanced_parts.append(f"- {len(self.ocr_tokens)} text regions detected")
            enhanced_parts.append("- Use OCR tokens to help identify text")
            if len(self.ocr_tokens) <= 10:
                enhanced_parts.append(f"- Detected text: {', '.join(self.ocr_tokens[:10])}")
        
        # 多答案提示
        if self.is_multi_answer:
            enhanced_parts.append("\n**Note**: Multiple valid answers may exist for this question.")
            enhanced_parts.append("Common variations include different spellings or phrasings.")
        
        # 答案格式指导
        if self.is_numeric:
            enhanced_parts.append("\n**Answer Format**: Provide the numeric value as shown in the image")
        elif self.question_type == 'yes_no':
            enhanced_parts.append("\n**Answer Format**: Answer with 'Yes' or 'No'")
        else:
            enhanced_parts.append("\n**Answer Format**: Read the text exactly as it appears")
        
        return "\n".join(enhanced_parts)
    
    def _classify_question_type(self) -> str:
        """分类问题类型"""
        if not self.question:
            return 'other'
        
        q_lower = self.question.lower()
        
        # 文本阅读
        if any(word in q_lower for word in ['read', 'say', 'says', 'written', 'text', 'word']):
            return 'text_reading'
        
        # 品牌识别
        if any(word in q_lower for word in ['brand', 'company', 'logo', 'manufacturer']):
            return 'brand_recognition'
        
        # 时间相关
        if any(word in q_lower for word in ['time', 'date', 'when', 'year', 'clock']):
            return 'temporal'
        
        # 数字相关
        if any(word in q_lower for word in ['price', 'cost', 'how much', 'number', 'phone']):
            return 'numeric'
        
        # 位置相关
        if any(word in q_lower for word in ['where', 'location', 'address', 'street']):
            return 'location'
        
        # 颜色相关
        if any(word in q_lower for word in ['color', 'colour']):
            return 'color'
        
        # 计数
        if any(word in q_lower for word in ['how many', 'count']):
            return 'counting'
        
        # Yes/No
        if q_lower.startswith(('is ', 'are ', 'does ', 'do ', 'can ', 'will ')):
            return 'yes_no'
        
        # What/Which
        if q_lower.startswith(('what', 'which')):
            return 'what_which'
        
        return 'other'
    
    def _is_numeric_answer(self, answer: str) -> bool:
        """检查答案是否为数值"""
        if not answer:
            return False
        
        answer_str = str(answer).strip()
        
        # 移除常见符号
        cleaned = answer_str.replace('$', '').replace(',', '').replace('%', '').strip()
        
        # 检查是否为数字
        try:
            float(cleaned)
            return True
        except:
            # 检查是否包含数字模式
            return bool(re.search(r'\d+\.?\d*', answer_str))
    
    def _check_answer_in_ocr(self) -> bool:
        """检查答案是否在OCR tokens中"""
        if not self.ocr_tokens or not self.answer:
            return False
        
        answer_lower = self.answer.lower()
        ocr_lower = [token.lower() for token in self.ocr_tokens]
        
        # 完全匹配
        if answer_lower in ocr_lower:
            return True
        
        # 部分匹配
        for token in ocr_lower:
            if answer_lower in token or token in answer_lower:
                return True
        
        return False
    
    def check_success(self, action: Any) -> Tuple[bool, str]:
        """
        检查Text-VQA答案
        
        支持多答案验证和灵活匹配
        """
        if not action:
            return False, "No answer provided"
        
        # 清理答案格式
        user_answer = str(action).strip()
        
        # 检查所有可能的答案
        for valid_answer in self.all_answers:
            valid_answer = str(valid_answer).strip()
            
            # 1. 完全匹配（忽略大小写）
            if user_answer.lower() == valid_answer.lower():
                return True, f"Correct! The answer is {valid_answer}"
            
            # 2. 包含匹配
            if valid_answer.lower() in user_answer.lower():
                # 检查是否只是多了一些解释
                if len(user_answer) <= len(valid_answer) * 2:
                    return True, f"Correct! The answer is {valid_answer}"
            
            # 3. 用户答案被包含（适用于缩写）
            if len(user_answer) >= 2 and user_answer.lower() in valid_answer.lower():
                return True, f"Correct! The answer is {valid_answer}"
        
        # 特殊类型检查
        primary_answer = self.answer
        
        # Yes/No答案
        if self.question_type == 'yes_no':
            user_yn = self._extract_yes_no(user_answer)
            if user_yn and user_yn == primary_answer.lower():
                return True, f"Correct! The answer is {primary_answer}"
        
        # 数值答案
        if self.is_numeric:
            success, message = self._check_numeric_answer(user_answer, primary_answer)
            if success:
                return success, message
        
        # 时间答案
        if self.question_type == 'temporal':
            success, message = self._check_temporal_answer(user_answer, primary_answer)
            if success:
                return success, message
        
        # 灵活文本匹配
        success, message = self._check_flexible_text(user_answer, primary_answer)
        if success:
            return success, message
        
        # 如果都不匹配，返回失败
        return False, f"Incorrect. The correct answer is {primary_answer}"
    
    def _extract_yes_no(self, text: str) -> Optional[str]:
        """从文本中提取Yes/No答案"""
        text_lower = text.lower().strip()
        
        if 'yes' in text_lower and 'no' not in text_lower:
            return 'yes'
        elif 'no' in text_lower and 'yes' not in text_lower:
            return 'no'
        
        return None
    
    def _check_numeric_answer(self, user_answer: str, correct_answer: str) -> Tuple[bool, str]:
        """检查数值答案"""
        user_num = self._extract_number(user_answer)
        correct_num = self._extract_number(correct_answer)
        
        if user_num is not None and correct_num is not None:
            if abs(user_num - correct_num) < 0.01:
                return True, f"Correct! The answer is {correct_answer}"
            
            # 允许小的相对误差
            if correct_num != 0:
                relative_error = abs(user_num - correct_num) / abs(correct_num)
                if relative_error <= 0.01:
                    return True, f"Correct! The answer is {correct_answer}"
        
        return False, "Numeric answer does not match"
    
    def _extract_number(self, text: str) -> Optional[float]:
        """从文本中提取数字"""
        if not text:
            return None
        
        # 清理文本
        text = text.strip()
        text = re.sub(r'[$,％%]', '', text)
        
        # 查找数字
        match = re.search(r'-?\d+\.?\d*', text)
        if match:
            try:
                return float(match.group())
            except:
                pass
        
        return None
    
    def _check_temporal_answer(self, user_answer: str, correct_answer: str) -> Tuple[bool, str]:
        """检查时间/日期答案"""
        # 标准化时间格式
        user_clean = re.sub(r'[:\s-/]', '', user_answer.lower())
        correct_clean = re.sub(r'[:\s-/]', '', correct_answer.lower())
        
        if user_clean == correct_clean:
            return True, f"Correct! The answer is {correct_answer}"
        
        # 检查包含关系
        if correct_clean in user_clean:
            return True, f"Correct! The answer is {correct_answer}"
        
        return False, "Time/date answer does not match"
    
    def _check_flexible_text(self, user_answer: str, correct_answer: str) -> Tuple[bool, str]:
        """灵活的文本匹配"""
        # 清理文本
        user_clean = self._clean_text(user_answer)
        correct_clean = self._clean_text(correct_answer)
        
        # 忽略标点和大小写
        user_words = user_clean.lower().split()
        correct_words = correct_clean.lower().split()
        
        # 如果是单词答案，检查核心词
        if len(correct_words) <= 2:
            for word in correct_words:
                if word in user_words:
                    return True, f"Correct! The answer is {correct_answer}"
        
        # 检查关键词匹配率
        if self._check_keyword_match(user_words, correct_words):
            return True, f"Correct! The answer is {correct_answer}"
        
        return False, "Answer does not match"
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除多余的标点和空格
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        return text.strip()
    
    def _check_keyword_match(self, user_words: List[str], correct_words: List[str]) -> bool:
        """检查关键词匹配"""
        if not correct_words:
            return False
        
        matches = sum(1 for word in correct_words if word in user_words)
        match_rate = matches / len(correct_words)
        
        return match_rate >= 0.5  # 50%以上匹配
    
    def validate(
        self,
        chat_history: List[Dict],
        observation: Any,
        full_history: Optional[List[Any]] = None
    ) -> Tuple[float, bool, str, Dict[str, Any]]:
        """
        验证Text-VQA任务执行情况
        """
        # 调用父类验证
        reward, done, message, info = super().validate(
            chat_history, observation, full_history
        )
        
        # 添加Text-VQA特定的信息
        info["task_type"] = self.task_type
        info["question_type"] = self.question_type
        info["has_ocr"] = self.has_ocr
        info["ocr_token_count"] = len(self.ocr_tokens)
        info["is_numeric"] = self.is_numeric
        info["is_multi_answer"] = self.is_multi_answer
        info["unique_answer_count"] = len(set(self.all_answers))
        
        # 分析OCR使用情况
        if self.has_ocr and info.get("answer_provided"):
            provided_answer = str(info["answer_provided"]).lower()
            ocr_used = any(token.lower() in provided_answer for token in self.ocr_tokens)
            info["ocr_tokens_used"] = ocr_used
        
        return reward, done, message, info
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取Text-VQA特定的指标"""
        metrics = super().get_metrics()
        
        metrics.update({
            "task_type": self.task_type,
            "question_type": self.question_type,
            "has_ocr": self.has_ocr,
            "ocr_token_count": len(self.ocr_tokens),
            "is_numeric": self.is_numeric,
            "is_multi_answer": self.is_multi_answer,
            "unique_answer_count": len(set(self.all_answers)),
            "text_complexity": self._assess_text_complexity(),
            "answer_in_ocr": self._check_answer_in_ocr()
        })
        
        # 添加OCR相关指标
        if self.has_ocr:
            metrics["ocr_tokens"] = self.ocr_tokens[:10]  # 前10个
        
        return metrics
    
    def _assess_text_complexity(self) -> str:
        """评估文本复杂度"""
        complexity_score = 0
        
        # 基于OCR token数量
        if len(self.ocr_tokens) > 20:
            complexity_score += 2
        elif len(self.ocr_tokens) > 10:
            complexity_score += 1
        
        # 基于答案长度
        if self.answer and len(self.answer.split()) > 3:
            complexity_score += 1
        
        # 基于多答案
        if self.is_multi_answer:
            complexity_score += 1
        
        # 返回复杂度等级
        if complexity_score >= 3:
            return "high"
        elif complexity_score >= 1:
            return "medium"
        else:
            return "low"
    
    def get_observation(self) -> Dict[str, Any]:
        """获取任务观察（覆盖父类方法以添加OCR信息）"""
        obs = super().get_observation()
        
        # 添加OCR信息
        if self.has_ocr:
            obs["ocr_available"] = True
            obs["ocr_token_count"] = len(self.ocr_tokens)
            # 可选：添加OCR tokens（如果需要）
            # obs["ocr_tokens"] = self.ocr_tokens
        
        return obs