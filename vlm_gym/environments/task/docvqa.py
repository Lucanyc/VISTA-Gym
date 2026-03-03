"""
DocVQA Task implementation for VLM Gym
Handles document-based visual question answering with answer localization
"""

from typing import Tuple, Dict, Any, List, Optional
import re
import json
from pathlib import Path

from .vision_qa_task import VisionQATask


class DocVQATask(VisionQATask):
    """
    DocVQA 特定任务
    
    专门处理文档相关的视觉问答任务，包括：
    - 文本提取（从文档中提取特定信息）
    - 数值识别（提取金额、日期、数量等）
    - 表格理解（从表格中提取数据）
    - 布局理解（理解文档结构和组织）
    - 答案定位（通过边界框定位答案位置）
    
    支持InfographicVQA等文档理解数据集
    """
    
    # 文档问题类型
    QUESTION_TYPES = {
        'entity_extraction': '实体提取（名称、地址、公司等）',
        'value_extraction': '数值提取（金额、百分比、数量等）', 
        'temporal': '时间信息（日期、时间、期限等）',
        'counting': '计数问题（数量统计）',
        'location': '位置查找（在文档中定位信息）',
        'comparison': '比较问题（比较文档中的数值或信息）',
        'yes_no': '是非判断',
        'other': '其他类型'
    }
    
    # 模板类型映射
    TEMPLATE_TYPES = {
        'reasoning': '需要推理的复杂问题',
        'data': '直接数据提取',
        'structure': '文档结构理解'
    }
    
    def __init__(self, task_id: str, adapter: Any):
        """
        初始化DocVQA任务
        
        Args:
            task_id: 任务ID
            adapter: DocVQA数据适配器
        """
        # 调用父类初始化
        super().__init__(task_id, adapter)
        
        # 初始化DocVQA特定属性
        self.template_id = None
        self.answer_bbox = []
        self.question_type = None
        self.answer_type = None
        self.has_bbox = False
        self.requires_ocr = True
        self.involves_table = False
        self.involves_form = False
        self.is_numeric = False
        self.is_temporal = False
        
        # 获取数据集名称
        task_data = adapter.get_task_data(task_id)
        self.dataset_name = task_data.get('dataset', 'docvqa')
    
    @classmethod
    def get_task_id(cls) -> str:
        """获取任务类型ID"""
        return "vlm-gym.docvqa"
    
    def setup(self) -> Tuple[str, Dict[str, Any]]:
        """设置DocVQA特定的任务"""
        # 调用父类setup
        task_goal, task_info = super().setup()
        
        # 从metadata中提取DocVQA特定信息
        self.template_id = self.metadata.get('template_id', 'unknown')
        self.answer_bbox = self.metadata.get('answer_bbox', [])
        self.original_question_id = self.metadata.get('original_question_id', -1)
        self.original_image_filename = self.metadata.get('original_image_filename', '')
        
        # 分析问题和答案特征
        self.question_type = self._classify_question_type()
        self.answer_type = self._classify_answer_type()
        self.has_bbox = len(self.answer_bbox) > 0
        self.requires_ocr = self._check_requires_ocr()
        self.involves_table = self._check_involves_table()
        self.involves_form = self._check_involves_form()
        self.is_numeric = self._is_numeric_answer()
        self.is_temporal = self._is_temporal_answer()
        
        # 添加DocVQA特定的处理
        task_info["template_id"] = self.template_id
        task_info["question_type"] = self.question_type
        task_info["answer_type"] = self.answer_type
        task_info["has_bbox"] = self.has_bbox
        task_info["requires_ocr"] = self.requires_ocr
        task_info["involves_table"] = self.involves_table
        task_info["involves_form"] = self.involves_form
        task_info["is_numeric"] = self.is_numeric
        task_info["is_temporal"] = self.is_temporal
        task_info["dataset"] = "docvqa"
        task_info["sub_dataset"] = self.dataset_name  # 可能是infographic_vqa等
        task_info["document_complexity"] = self._assess_document_complexity()
        task_info["answer_complexity"] = self._assess_answer_complexity()
        
        # 如果有边界框信息，添加到task_info
        if self.has_bbox:
            task_info["answer_bbox"] = self.answer_bbox
            task_info["bbox_area"] = self._calculate_bbox_area()
        
        # 修改任务目标以包含DocVQA特定指导
        enhanced_goal = self._enhance_task_goal(task_goal)
        
        return enhanced_goal, task_info
    
    def _enhance_task_goal(self, base_goal: str) -> str:
        """增强任务目标描述，添加DocVQA特定的指导"""
        enhanced_parts = [base_goal]
        
        # 添加文档分析通用指导
        enhanced_parts.append("\n**Document Analysis Guidelines:**")
        enhanced_parts.append("1. Carefully examine the entire document layout")
        enhanced_parts.append("2. Pay attention to headers, titles, and section organization")
        enhanced_parts.append("3. Look for relevant text, numbers, tables, or forms")
        enhanced_parts.append("4. Extract information precisely as it appears in the document")
        
        # 根据问题类型添加特定指导
        if self.question_type == 'entity_extraction':
            enhanced_parts.append("\n**Entity Extraction Tips:**")
            enhanced_parts.append("- Look for proper nouns, company names, addresses")
            enhanced_parts.append("- Check headers, letterheads, and signature areas")
            enhanced_parts.append("- Be exact with spelling and capitalization")
            
        elif self.question_type == 'value_extraction':
            enhanced_parts.append("\n**Value Extraction Tips:**")
            enhanced_parts.append("- Look for numbers, percentages, currency symbols")
            enhanced_parts.append("- Check tables, invoices, or financial sections")
            enhanced_parts.append("- Include units or currency symbols if present")
            
        elif self.question_type == 'temporal':
            enhanced_parts.append("\n**Date/Time Extraction Tips:**")
            enhanced_parts.append("- Look for date formats (MM/DD/YYYY, DD-MM-YY, etc.)")
            enhanced_parts.append("- Check document headers, footers, or date fields")
            enhanced_parts.append("- Maintain the exact format from the document")
            
        elif self.question_type == 'counting':
            enhanced_parts.append("\n**Counting Tips:**")
            enhanced_parts.append("- Carefully count all relevant items")
            enhanced_parts.append("- Check tables, lists, or repeated elements")
            enhanced_parts.append("- Provide the exact count as a number")
        
        # 根据文档类型添加指导
        if self.involves_table:
            enhanced_parts.append("\n**Table Reading Tips:**")
            enhanced_parts.append("- Identify the relevant table")
            enhanced_parts.append("- Understand column/row headers")
            enhanced_parts.append("- Locate the specific cell or cells needed")
            
        if self.involves_form:
            enhanced_parts.append("\n**Form Reading Tips:**")
            enhanced_parts.append("- Look for labeled fields and their values")
            enhanced_parts.append("- Check boxes, filled fields, and selections")
            enhanced_parts.append("- Pay attention to form structure")
        
        # 根据模板类型添加指导
        if self.template_id == 'reasoning':
            enhanced_parts.append("\n**This question requires reasoning:**")
            enhanced_parts.append("- You may need to combine multiple pieces of information")
            enhanced_parts.append("- Think step-by-step through the logic")
            enhanced_parts.append("- Show your reasoning process")
            
        elif self.template_id == 'data':
            enhanced_parts.append("\n**Direct data extraction required:**")
            enhanced_parts.append("- Find and extract the exact information requested")
            enhanced_parts.append("- No interpretation needed, just accurate extraction")
        
        # 添加答案格式指导
        if self.is_numeric:
            enhanced_parts.append("\n**Answer Format**: Provide the numeric value (with units if applicable)")
        elif self.is_temporal:
            enhanced_parts.append("\n**Answer Format**: Provide the date/time in the exact format from the document")
        elif self.answer_type == 'yes_no':
            enhanced_parts.append("\n**Answer Format**: Answer with 'Yes' or 'No'")
        else:
            enhanced_parts.append("\n**Answer Format**: Extract the text exactly as it appears in the document")
        
        # 如果有边界框，提醒注意特定区域
        if self.has_bbox:
            enhanced_parts.append("\n**Note**: The answer can be found in a specific region of the document.")
        
        return "\n".join(enhanced_parts)
    
    def _classify_question_type(self) -> str:
        """分类问题类型"""
        if not self.question:
            return 'other'
        
        q_lower = self.question.lower()
        
        # 实体提取
        entity_keywords = ['who', 'what company', 'what organization', 'name', 
                          'address', 'which company', 'which organization']
        if any(kw in q_lower for kw in entity_keywords):
            return 'entity_extraction'
        
        # 数值提取
        value_keywords = ['how much', 'how many', 'what is the value', 'total', 
                         'amount', 'price', 'cost', 'percentage', 'rate']
        if any(kw in q_lower for kw in value_keywords):
            return 'value_extraction'
        
        # 时间信息
        temporal_keywords = ['when', 'what date', 'what time', 'what year', 
                           'what month', 'deadline', 'due date', 'expiry']
        if any(kw in q_lower for kw in temporal_keywords):
            return 'temporal'
        
        # 计数
        if 'count' in q_lower or 'number of' in q_lower:
            return 'counting'
        
        # 位置
        if 'where' in q_lower or 'location' in q_lower or 'find' in q_lower:
            return 'location'
        
        # 比较
        comparison_keywords = ['compare', 'difference', 'higher', 'lower', 
                             'more', 'less', 'greater', 'smaller']
        if any(kw in q_lower for kw in comparison_keywords):
            return 'comparison'
        
        # Yes/No
        if q_lower.startswith(('is ', 'are ', 'does ', 'do ', 'can ', 'will ', 'has ', 'have ')):
            return 'yes_no'
        
        return 'other'
    
    def _classify_answer_type(self) -> str:
        """分类答案类型"""
        if not self.answer:
            return 'unknown'
        
        answer_str = str(self.answer).strip().lower()
        
        # Yes/No
        if answer_str in ['yes', 'no']:
            return 'yes_no'
        
        # 数字
        if self._is_numeric_answer():
            return 'numeric'
        
        # 日期/时间
        if self._is_temporal_answer():
            return 'temporal'
        
        # 单词
        if len(answer_str.split()) == 1:
            return 'single_word'
        
        # 短语
        if len(answer_str.split()) <= 3:
            return 'phrase'
        
        # 句子或长文本
        return 'sentence'
    
    def _check_requires_ocr(self) -> bool:
        """检查是否需要OCR（文本识别）"""
        # DocVQA任务通常都需要OCR
        return True
    
    def _check_involves_table(self) -> bool:
        """检查是否涉及表格"""
        if not self.question:
            return False
        
        q_lower = self.question.lower()
        table_keywords = ['table', 'row', 'column', 'cell', 'grid']
        return any(kw in q_lower for kw in table_keywords)
    
    def _check_involves_form(self) -> bool:
        """检查是否涉及表单"""
        if not self.question:
            return False
        
        q_lower = self.question.lower()
        form_keywords = ['form', 'field', 'checkbox', 'fill', 'entry', 'input']
        return any(kw in q_lower for kw in form_keywords)
    
    def _is_numeric_answer(self) -> bool:
        """检查答案是否为数值"""
        if not self.answer:
            return False
        
        answer_str = str(self.answer).strip()
        
        # 移除常见的货币符号和单位
        cleaned = answer_str.replace('$', '').replace(',', '').replace('%', '')
        cleaned = cleaned.replace('€', '').replace('£', '').replace('¥', '')
        
        # 检查是否为数字
        try:
            float(cleaned)
            return True
        except:
            # 检查是否包含数字模式
            return bool(re.search(r'\d+\.?\d*', answer_str))
    
    def _is_temporal_answer(self) -> bool:
        """检查答案是否为时间/日期"""
        if not self.answer:
            return False
        
        answer_str = str(self.answer).strip()
        
        # 常见的日期模式
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # MM/DD/YYYY or DD-MM-YY
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',     # YYYY-MM-DD
            r'\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',  # DD Month
            r'(January|February|March|April|May|June|July|August|September|October|November|December)',
            r'\d{1,2}:\d{2}',  # Time HH:MM
        ]
        
        for pattern in date_patterns:
            if re.search(pattern, answer_str, re.IGNORECASE):
                return True
        
        return False
    
    def _calculate_bbox_area(self) -> float:
        """计算边界框面积"""
        if len(self.answer_bbox) >= 4:
            # 假设格式为 [x, y, width, height]
            return self.answer_bbox[2] * self.answer_bbox[3]
        return 0.0
    
    def check_success(self, action: Any) -> Tuple[bool, str]:
        """
        检查DocVQA答案
        
        根据答案类型采用不同的匹配策略
        """
        if not action:
            return False, "No answer provided"
        
        # 清理答案格式
        user_answer = str(action).strip()
        correct_answer = str(self.answer).strip()
        
        # 1. Yes/No答案
        if self.answer_type == 'yes_no':
            user_yn = self._extract_yes_no(user_answer)
            correct_yn = correct_answer.lower()
            
            if user_yn and user_yn == correct_yn:
                return True, f"Correct! The answer is {correct_answer}"
            elif user_yn:
                return False, f"Incorrect. The correct answer is {correct_answer}"
            else:
                return False, "Please provide a clear Yes or No answer"
        
        # 2. 数值答案
        if self.is_numeric:
            success, message = self._check_numeric_answer(user_answer, correct_answer)
            if success or message != "Not a numeric comparison":
                return success, message
        
        # 3. 时间/日期答案
        if self.is_temporal:
            success, message = self._check_temporal_answer(user_answer, correct_answer)
            if success:
                return success, message
        
        # 4. 文本答案 - 使用灵活匹配
        success, message = self._check_text_answer(user_answer, correct_answer)
        return success, message
    
    def _extract_yes_no(self, text: str) -> Optional[str]:
        """从文本中提取Yes/No答案"""
        text_lower = text.lower().strip()
        
        # 直接匹配
        if text_lower in ['yes', 'no']:
            return text_lower
        
        # 包含yes/no的判断
        if text_lower.startswith('yes') and 'no' not in text_lower:
            return 'yes'
        elif text_lower.startswith('no') and 'yes' not in text_lower:
            return 'no'
        
        # 在句子中查找
        if ' yes' in text_lower or 'yes,' in text_lower or 'yes.' in text_lower:
            return 'yes'
        elif ' no' in text_lower or 'no,' in text_lower or 'no.' in text_lower:
            return 'no'
        
        return None
    
    def _check_numeric_answer(self, user_answer: str, correct_answer: str) -> Tuple[bool, str]:
        """检查数值答案"""
        # 提取数字
        user_num = self._extract_number(user_answer)
        correct_num = self._extract_number(correct_answer)
        
        if user_num is not None and correct_num is not None:
            # 完全匹配
            if user_num == correct_num:
                return True, f"Correct! The answer is {correct_answer}"
            
            # 允许小的相对误差（对于可能的OCR错误）
            if correct_num != 0:
                relative_error = abs(user_num - correct_num) / abs(correct_num)
                if relative_error <= 0.01:  # 1%误差
                    return True, f"Correct! (within acceptable range of {correct_answer})"
            
            # 检查是否是单位问题（如1000 vs 1k）
            if self._check_unit_conversion(user_num, correct_num):
                return True, f"Correct! The answer is {correct_answer}"
            
            return False, f"Incorrect. The correct answer is {correct_answer}"
        
        return False, "Not a numeric comparison"
    
    def _extract_number(self, text: str) -> Optional[float]:
        """从文本中提取数字"""
        if not text:
            return None
        
        # 清理文本
        text = text.strip()
        
        # 移除货币符号
        text = text.replace('$', '').replace('€', '').replace('£', '').replace('¥', '')
        
        # 移除千位分隔符
        text = text.replace(',', '')
        
        # 处理百分号
        is_percentage = '%' in text
        text = text.replace('%', '')
        
        # 查找数字模式
        number_pattern = r'-?\d+\.?\d*'
        matches = re.findall(number_pattern, text)
        
        if matches:
            try:
                number = float(matches[0])
                if is_percentage:
                    number = number / 100
                return number
            except:
                pass
        
        return None
    
    def _check_unit_conversion(self, num1: float, num2: float) -> bool:
        """检查是否是单位转换问题"""
        # 常见的转换比例
        ratios = [1000, 100, 12, 10]  # K/M, 百分比, 月/年等
        
        for ratio in ratios:
            if abs(num1 * ratio - num2) < 0.01 or abs(num1 / ratio - num2) < 0.01:
                return True
        
        return False
    
    def _check_temporal_answer(self, user_answer: str, correct_answer: str) -> Tuple[bool, str]:
        """检查时间/日期答案"""
        # 标准化日期格式
        user_normalized = self._normalize_date(user_answer)
        correct_normalized = self._normalize_date(correct_answer)
        
        if user_normalized and correct_normalized:
            if user_normalized == correct_normalized:
                return True, f"Correct! The answer is {correct_answer}"
        
        # 直接文本匹配（保留原始格式）
        if user_answer.lower() == correct_answer.lower():
            return True, f"Correct! The answer is {correct_answer}"
        
        # 检查是否包含正确答案
        if correct_answer.lower() in user_answer.lower():
            return True, f"Correct! The answer is {correct_answer}"
        
        return False, f"Incorrect. The correct answer is {correct_answer}"
    
    def _normalize_date(self, date_str: str) -> Optional[str]:
        """标准化日期格式"""
        # 这里可以添加更复杂的日期解析逻辑
        # 简单实现：移除常见的分隔符
        if date_str:
            normalized = date_str.replace('/', '-').replace('.', '-').replace(' ', '-')
            return normalized.lower()
        return None
    
    def _check_text_answer(self, user_answer: str, correct_answer: str) -> Tuple[bool, str]:
        """检查文本答案 - 使用灵活的匹配策略"""
        # 清理和标准化
        user_clean = self._clean_text(user_answer)
        correct_clean = self._clean_text(correct_answer)
        
        # 1. 完全匹配
        if user_clean == correct_clean:
            return True, f"Correct! The answer is {correct_answer}"
        
        # 2. 忽略大小写匹配
        if user_clean.lower() == correct_clean.lower():
            return True, f"Correct! The answer is {correct_answer}"
        
        # 3. 包含匹配（用户答案包含正确答案）
        if correct_clean.lower() in user_clean.lower():
            # 检查是否只是多了一些解释
            if len(user_clean) <= len(correct_clean) * 2:
                return True, f"Correct! The answer is {correct_answer}"
        
        # 4. 核心词匹配（适用于短语答案）
        if self.answer_type in ['phrase', 'sentence']:
            if self._check_key_words_match(user_clean, correct_clean):
                return True, f"Correct! The answer is {correct_answer}"
        
        # 5. 编辑距离检查（拼写错误容忍）
        if self._check_edit_distance(user_clean.lower(), correct_clean.lower()):
            return True, f"Correct! The answer is {correct_answer}"
        
        return False, f"Incorrect. The correct answer is {correct_answer}"
    
    def _clean_text(self, text: str) -> str:
        """清理文本 - 移除多余的空格和标点"""
        # 移除多余空格
        text = ' '.join(text.split())
        
        # 移除首尾的标点
        text = text.strip('.,!?;:')
        
        return text.strip()
    
    def _check_key_words_match(self, user_text: str, correct_text: str) -> bool:
        """检查关键词匹配"""
        # 提取关键词（去除常见停用词）
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'of', 'in', 'on', 'at', 'to', 'for'}
        
        user_words = set(w.lower() for w in user_text.split() if w.lower() not in stop_words)
        correct_words = set(w.lower() for w in correct_text.split() if w.lower() not in stop_words)
        
        # 如果没有关键词，返回False
        if not correct_words:
            return False
        
        # 计算匹配率
        matches = user_words & correct_words
        match_rate = len(matches) / len(correct_words)
        
        return match_rate >= 0.8  # 80%的关键词匹配
    
    def _check_edit_distance(self, s1: str, s2: str, max_distance: int = 2) -> bool:
        """检查编辑距离（简单实现）"""
        # 对于短答案，允许1-2个字符的差异
        if abs(len(s1) - len(s2)) > max_distance:
            return False
        
        # 这里可以实现更复杂的编辑距离算法
        # 简单检查：字符差异
        diff_count = sum(1 for c1, c2 in zip(s1, s2) if c1 != c2)
        diff_count += abs(len(s1) - len(s2))
        
        return diff_count <= max_distance
    
    def validate(
        self,
        chat_history: List[Dict],
        observation: Any,
        full_history: Optional[List[Any]] = None
    ) -> Tuple[float, bool, str, Dict[str, Any]]:
        """
        验证DocVQA任务执行情况
        """
        # 调用父类验证
        reward, done, message, info = super().validate(
            chat_history, observation, full_history
        )
        
        # 添加DocVQA特定的信息
        info["template_id"] = self.template_id
        info["question_type"] = self.question_type
        info["answer_type"] = self.answer_type
        info["has_bbox"] = self.has_bbox
        info["requires_ocr"] = self.requires_ocr
        info["involves_table"] = self.involves_table
        info["involves_form"] = self.involves_form
        info["is_numeric"] = self.is_numeric
        info["is_temporal"] = self.is_temporal
        
        # 如果有边界框，添加相关信息
        if self.has_bbox:
            info["answer_bbox"] = self.answer_bbox
            info["bbox_area"] = self._calculate_bbox_area()
        
        # 分析答案质量
        if info.get("answer_provided"):
            provided_answer = str(info["answer_provided"])
            
            # 检查答案长度
            answer_length = len(provided_answer.split())
            info["answer_word_count"] = answer_length
            
            # 检查是否过于冗长
            if answer_length > 50 and self.answer_type in ['single_word', 'phrase']:
                info["verbose_answer"] = True
            
            # 对于数值答案，提取数值
            if self.is_numeric:
                extracted_num = self._extract_number(provided_answer)
                if extracted_num is not None:
                    info["extracted_number"] = extracted_num
        
        return reward, done, message, info
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取DocVQA特定的指标"""
        metrics = super().get_metrics()
        
        metrics.update({
            "template_id": self.template_id,
            "question_type": self.question_type,
            "answer_type": self.answer_type,
            "has_bbox": self.has_bbox,
            "requires_ocr": self.requires_ocr,
            "involves_table": self.involves_table,
            "involves_form": self.involves_form,
            "is_numeric": self.is_numeric,
            "is_temporal": self.is_temporal,
            "document_complexity": self._assess_document_complexity(),
            "answer_complexity": self._assess_answer_complexity()
        })
        
        # 如果有边界框，添加面积信息
        if self.has_bbox:
            metrics["bbox_area"] = self._calculate_bbox_area()
        
        return metrics
    
    def _assess_document_complexity(self) -> str:
        """评估文档复杂度"""
        complexity_factors = 0
        
        # 基于问题类型
        if self.question_type in ['comparison', 'counting']:
            complexity_factors += 1
        
        # 基于文档元素
        if self.involves_table:
            complexity_factors += 1
        if self.involves_form:
            complexity_factors += 1
        
        # 基于模板
        if self.template_id == 'reasoning':
            complexity_factors += 2
        
        # 返回复杂度等级
        if complexity_factors >= 3:
            return "high"
        elif complexity_factors >= 1:
            return "medium"
        else:
            return "low"
    
    def _assess_answer_complexity(self) -> str:
        """评估答案复杂度"""
        if self.answer_type in ['single_word', 'yes_no']:
            return "low"
        elif self.answer_type in ['phrase', 'numeric', 'temporal']:
            return "medium"
        else:
            return "high"
    
    def get_document_regions(self) -> Dict[str, str]:
        """获取文档区域提示（辅助方法）"""
        regions = {
            "header": "文档顶部，通常包含标题、日期、编号",
            "body": "文档主体，包含主要内容",
            "table": "表格区域，结构化数据",
            "form": "表单区域，字段和填写内容",
            "footer": "文档底部，可能包含页码、签名等"
        }
        return regions
    
    def get_extraction_tips(self) -> List[str]:
        """获取信息提取提示"""
        tips = []
        
        if self.is_numeric:
            tips.append("注意数字的完整性，包括小数点和单位")
        
        if self.is_temporal:
            tips.append("保持日期格式与文档中一致")
        
        if self.involves_table:
            tips.append("仔细定位表格中的行列交叉点")
        
        if self.has_bbox:
            tips.append("答案位于文档的特定区域")
        
        return tips
    
    def analyze_bbox_coverage(self, task_id: str) -> Dict[str, Any]:
        """分析边界框覆盖情况"""
        # 这是测试脚本中用到的方法
        return {
            'task_id': task_id,
            'has_bbox': self.has_bbox,
            'bbox': self.answer_bbox,
            'bbox_area': self._calculate_bbox_area() if self.answer_bbox else 0,
            'answer': self.answer,
            'question': self.question,
            'template': self.template_id
        }