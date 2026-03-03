"""
DVQA Task implementation for VLM Gym
Handles diagram/chart-based visual question answering
"""

from typing import Tuple, Dict, Any, List, Optional
import re
import json
from pathlib import Path

from .vision_qa_task import VisionQATask


class DVQATask(VisionQATask):
    """
    DVQA (Diagram VQA) 特定任务
    
    专门处理图表相关的视觉问答任务，包括：
    - 数值提取（从图表中提取具体数值）
    - 标签识别（识别轴标签、图例等）
    - 结构理解（理解图表类型和组成）
    - 比较分析（比较图表中的数据）
    - 颜色识别（识别图表元素的颜色）
    - 推理任务（基于图表数据进行推理）
    
    支持bar charts, line graphs, pie charts等多种图表类型
    """
    
    # 图表问题类型
    QUESTION_TYPES = {
        'value_extraction': '数值提取（具体数值、百分比等）',
        'label_reading': '标签阅读（轴标签、图例、标题等）',
        'comparison': '比较问题（最大、最小、差异等）',
        'counting': '计数问题（柱子数量、数据点等）',
        'color': '颜色识别（元素颜色）',
        'structure': '结构理解（图表类型、布局等）',
        'chart_element': '图表元素（识别特定元素）',
        'yes_no': '是非判断',
        'entity_extraction': '实体提取（名称、类别等）',
        'other': '其他类型'
    }
    
    # 模板类型映射
    TEMPLATE_TYPES = {
        'reasoning': '需要推理的复杂问题（比较、计算、分析）',
        'data': '直接数据提取（读取具体值）',
        'structure': '图表结构理解（类型、布局、元素）'
    }
    
    # 图表类型
    CHART_TYPES = {
        'bar': '条形图/柱状图',
        'line': '折线图',
        'pie': '饼图',
        'scatter': '散点图',
        'area': '面积图',
        'unknown': '未知类型'
    }
    
    def __init__(self, task_id: str, adapter: Any):
        """
        初始化DVQA任务
        
        Args:
            task_id: 任务ID
            adapter: DVQA数据适配器
        """
        # 调用父类初始化
        super().__init__(task_id, adapter)
        
        # 初始化DVQA特定属性
        self.template_id = None
        self.answer_bbox = []
        self.question_type = None
        self.answer_type = None
        self.chart_type = None
        self.has_bbox = False
        self.is_numeric = False
        self.is_color_question = False
        self.is_comparison = False
        self.involves_counting = False
        self.single_word_answer = False
        
        # 获取数据集名称
        task_data = adapter.get_task_data(task_id)
        self.dataset_name = task_data.get('dataset', 'dvqa')
    
    @classmethod
    def get_task_id(cls) -> str:
        """获取任务类型ID"""
        return "vlm-gym.dvqa"
    
    def setup(self) -> Tuple[str, Dict[str, Any]]:
        """设置DVQA特定的任务"""
        # 调用父类setup
        task_goal, task_info = super().setup()
        
        # 从metadata中提取DVQA特定信息
        self.template_id = self.metadata.get('template_id', 'unknown')
        self.answer_bbox = self.metadata.get('answer_bbox', [])
        self.original_question_id = self.metadata.get('original_question_id', -1)
        self.original_image_filename = self.metadata.get('original_image_filename', '')
        
        # 从图片名称提取图表类型
        self.chart_type = self._extract_chart_type()
        
        # 分析问题和答案特征
        self.question_type = self._classify_question_type()
        self.answer_type = self._classify_answer_type()
        self.has_bbox = len(self.answer_bbox) > 0
        self.is_numeric = self._is_numeric_answer()
        self.is_color_question = self._is_color_question()
        self.is_comparison = self._is_comparison_question()
        self.involves_counting = self._involves_counting()
        self.single_word_answer = len(str(self.answer).split()) == 1
        
        # 添加DVQA特定的处理
        task_info["template_id"] = self.template_id
        task_info["question_type"] = self.question_type
        task_info["answer_type"] = self.answer_type
        task_info["chart_type"] = self.chart_type
        task_info["has_bbox"] = self.has_bbox
        task_info["is_numeric"] = self.is_numeric
        task_info["is_color_question"] = self.is_color_question
        task_info["is_comparison"] = self.is_comparison
        task_info["involves_counting"] = self.involves_counting
        task_info["single_word_answer"] = self.single_word_answer
        task_info["dataset"] = "dvqa"
        task_info["chart_complexity"] = self._assess_chart_complexity()
        task_info["answer_complexity"] = self._assess_answer_complexity()
        
        # 如果有边界框信息，添加到task_info
        if self.has_bbox:
            task_info["answer_bbox"] = self.answer_bbox
            task_info["bbox_area"] = self._calculate_bbox_area()
        
        # 修改任务目标以包含DVQA特定指导
        enhanced_goal = self._enhance_task_goal(task_goal)
        
        return enhanced_goal, task_info
    
    def _extract_chart_type(self) -> str:
        """从图片文件名提取图表类型"""
        if self.original_image_filename:
            parts = self.original_image_filename.split('_')
            if parts:
                chart_type = parts[0].lower()
                if chart_type in self.CHART_TYPES:
                    return chart_type
        return 'unknown'
    
    def _enhance_task_goal(self, base_goal: str) -> str:
        """增强任务目标描述，添加DVQA特定的指导"""
        enhanced_parts = [base_goal]
        
        # 添加图表分析通用指导
        enhanced_parts.append("\n**Chart Analysis Guidelines:**")
        enhanced_parts.append("1. Identify the chart type and structure")
        enhanced_parts.append("2. Locate axes labels, legends, and titles")
        enhanced_parts.append("3. Understand the data representation method")
        enhanced_parts.append("4. Extract information precisely")
        
        # 根据图表类型添加特定指导
        if self.chart_type == 'bar':
            enhanced_parts.append("\n**Bar Chart Tips:**")
            enhanced_parts.append("- Check if bars are horizontal or vertical")
            enhanced_parts.append("- Read values from bar heights/lengths")
            enhanced_parts.append("- Pay attention to the scale on axes")
            enhanced_parts.append("- Identify bar labels and groupings")
            
        elif self.chart_type == 'line':
            enhanced_parts.append("\n**Line Chart Tips:**")
            enhanced_parts.append("- Follow the line trajectory")
            enhanced_parts.append("- Identify data points and their values")
            enhanced_parts.append("- Check for multiple lines and legends")
            enhanced_parts.append("- Note trends and patterns")
            
        elif self.chart_type == 'pie':
            enhanced_parts.append("\n**Pie Chart Tips:**")
            enhanced_parts.append("- Identify slice sizes and proportions")
            enhanced_parts.append("- Check for percentage labels")
            enhanced_parts.append("- Match colors to legend entries")
            enhanced_parts.append("- Consider the whole as 100%")
        
        # 根据问题类型添加特定指导
        if self.question_type == 'value_extraction':
            enhanced_parts.append("\n**Value Extraction Tips:**")
            enhanced_parts.append("- Locate the specific data point")
            enhanced_parts.append("- Read the exact value from the chart")
            enhanced_parts.append("- Pay attention to units and scale")
            
        elif self.question_type == 'comparison':
            enhanced_parts.append("\n**Comparison Tips:**")
            enhanced_parts.append("- Identify all relevant data points")
            enhanced_parts.append("- Compare values systematically")
            enhanced_parts.append("- Determine the maximum/minimum/difference")
            enhanced_parts.append("- Be precise with comparisons")
            
        elif self.question_type == 'counting':
            enhanced_parts.append("\n**Counting Tips:**")
            enhanced_parts.append("- Count all relevant elements carefully")
            enhanced_parts.append("- Don't miss grouped or stacked elements")
            enhanced_parts.append("- Provide the exact count")
            
        elif self.is_color_question:
            enhanced_parts.append("\n**Color Identification Tips:**")
            enhanced_parts.append("- Identify the specific element in question")
            enhanced_parts.append("- Determine its color precisely")
            enhanced_parts.append("- Use standard color names")
        
        # 根据模板类型添加指导
        if self.template_id == 'reasoning':
            enhanced_parts.append("\n**This is a reasoning question:**")
            enhanced_parts.append("- Analyze the chart data comprehensively")
            enhanced_parts.append("- Apply logical reasoning to find the answer")
            enhanced_parts.append("- Consider relationships between data points")
            
        elif self.template_id == 'data':
            enhanced_parts.append("\n**Direct data extraction required:**")
            enhanced_parts.append("- Find the specific data point requested")
            enhanced_parts.append("- Extract the exact value without interpretation")
            
        elif self.template_id == 'structure':
            enhanced_parts.append("\n**Chart structure analysis required:**")
            enhanced_parts.append("- Understand the chart organization")
            enhanced_parts.append("- Identify structural elements")
            enhanced_parts.append("- Analyze layout and design features")
        
        # 添加答案格式指导
        if self.is_numeric:
            enhanced_parts.append("\n**Answer Format**: Provide the numeric value only")
        elif self.is_color_question:
            enhanced_parts.append("\n**Answer Format**: Provide the color name (e.g., red, blue, green)")
        elif self.answer_type == 'yes_no':
            enhanced_parts.append("\n**Answer Format**: Answer with 'yes' or 'no'")
        elif self.single_word_answer:
            enhanced_parts.append("\n**Answer Format**: Provide a single word answer")
        else:
            enhanced_parts.append("\n**Answer Format**: Keep your answer concise and precise")
        
        # 如果有边界框，提醒注意特定区域
        if self.has_bbox:
            enhanced_parts.append("\n**Note**: The answer can be found in a specific region of the chart.")
        
        return "\n".join(enhanced_parts)
    
    def _classify_question_type(self) -> str:
        """分类问题类型"""
        if not self.question:
            return 'other'
        
        q_lower = self.question.lower()
        
        # 数值提取
        value_keywords = ['what is the value', 'how much', 'how many', 'what value',
                         'what number', 'what percentage', 'what is the amount']
        if any(kw in q_lower for kw in value_keywords):
            return 'value_extraction'
        
        # 标签阅读
        label_keywords = ['what is the label', 'what is the title', 'axis label',
                         'legend', 'what does', 'name of']
        if any(kw in q_lower for kw in label_keywords):
            return 'label_reading'
        
        # 比较
        comparison_keywords = ['largest', 'smallest', 'highest', 'lowest', 'maximum',
                             'minimum', 'more than', 'less than', 'greater', 'compare']
        if any(kw in q_lower for kw in comparison_keywords):
            return 'comparison'
        
        # 计数
        if 'how many' in q_lower and any(word in q_lower for word in ['bars', 'lines', 'points', 'groups', 'categories']):
            return 'counting'
        
        # 颜色
        if 'color' in q_lower or 'colour' in q_lower:
            return 'color'
        
        # 结构
        structure_keywords = ['horizontal', 'vertical', 'type of chart', 'pattern',
                            'grouped', 'stacked', 'layout']
        if any(kw in q_lower for kw in structure_keywords):
            return 'structure'
        
        # 图表元素
        if any(word in q_lower for word in ['bar', 'line', 'slice', 'point', 'segment']):
            return 'chart_element'
        
        # Yes/No
        if q_lower.startswith(('is ', 'are ', 'does ', 'do ', 'can ', 'will ', 'has ', 'have ')):
            return 'yes_no'
        
        # 实体提取
        if any(word in q_lower for word in ['what', 'which', 'name']):
            return 'entity_extraction'
        
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
        
        # 颜色
        colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'black', 
                 'white', 'gray', 'grey', 'brown', 'pink', 'cyan', 'magenta']
        if answer_str in colors:
            return 'color'
        
        # 单词
        if len(answer_str.split()) == 1:
            return 'single_word'
        
        # 短语
        if len(answer_str.split()) <= 3:
            return 'phrase'
        
        # 句子或长文本
        return 'sentence'
    
    def _is_numeric_answer(self) -> bool:
        """检查答案是否为数值"""
        if not self.answer:
            return False
        
        answer_str = str(self.answer).strip()
        
        # 移除常见的单位和符号
        cleaned = answer_str.replace('$', '').replace(',', '').replace('%', '')
        
        # 检查是否为数字
        try:
            float(cleaned)
            return True
        except:
            # 检查是否包含数字模式
            return bool(re.search(r'^\d+\.?\d*$', cleaned))
    
    def _is_color_question(self) -> bool:
        """检查是否为颜色相关问题"""
        if not self.question:
            return False
        return 'color' in self.question.lower() or 'colour' in self.question.lower()
    
    def _is_comparison_question(self) -> bool:
        """检查是否为比较类问题"""
        if not self.question:
            return False
        
        q_lower = self.question.lower()
        comparison_words = ['largest', 'smallest', 'highest', 'lowest', 'maximum',
                          'minimum', 'more', 'less', 'greater', 'compare', 'difference']
        return any(word in q_lower for word in comparison_words)
    
    def _involves_counting(self) -> bool:
        """检查是否涉及计数"""
        if not self.question:
            return False
        
        q_lower = self.question.lower()
        return 'how many' in q_lower or 'count' in q_lower or 'number of' in q_lower
    
    def _calculate_bbox_area(self) -> float:
        """计算边界框面积"""
        if len(self.answer_bbox) >= 4:
            # 假设格式为 [x, y, width, height]
            return self.answer_bbox[2] * self.answer_bbox[3]
        return 0.0
    
    def check_success(self, action: Any) -> Tuple[bool, str]:
        """
        检查DVQA答案
        
        针对图表问答的特点采用合适的匹配策略
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
                return False, "Please provide a clear yes or no answer"
        
        # 2. 数值答案 - DVQA中很常见
        if self.is_numeric:
            success, message = self._check_numeric_answer(user_answer, correct_answer)
            if success or message != "Not a numeric comparison":
                return success, message
        
        # 3. 颜色答案
        if self.answer_type == 'color':
            success, message = self._check_color_answer(user_answer, correct_answer)
            if success:
                return success, message
        
        # 4. 文本答案 - 使用严格匹配（DVQA答案通常很精确）
        success, message = self._check_text_answer(user_answer, correct_answer)
        return success, message
    
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
        
        return None
    
    def _check_numeric_answer(self, user_answer: str, correct_answer: str) -> Tuple[bool, str]:
        """检查数值答案 - DVQA特别重要"""
        # 提取数字
        user_num = self._extract_number(user_answer)
        correct_num = self._extract_number(correct_answer)
        
        if user_num is not None and correct_num is not None:
            # 完全匹配
            if user_num == correct_num:
                return True, f"Correct! The answer is {correct_answer}"
            
            # 对于整数答案，要求完全匹配
            if correct_num == int(correct_num):
                return False, f"Incorrect. The correct answer is {correct_answer}"
            
            # 对于小数，允许极小的误差
            if abs(user_num - correct_num) < 0.01:
                return True, f"Correct! The answer is {correct_answer}"
            
            return False, f"Incorrect. The correct answer is {correct_answer}"
        
        return False, "Not a numeric comparison"
    
    def _extract_number(self, text: str) -> Optional[float]:
        """从文本中提取数字"""
        if not text:
            return None
        
        # 清理文本
        text = text.strip()
        
        # 查找数字模式
        number_pattern = r'-?\d+\.?\d*'
        matches = re.findall(number_pattern, text)
        
        if matches:
            try:
                return float(matches[0])
            except:
                pass
        
        # 处理英文数字
        number_words = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
            'ten': 10, 'eleven': 11, 'twelve': 12
        }
        
        text_lower = text.lower()
        for word, value in number_words.items():
            if word in text_lower:
                return float(value)
        
        return None
    
    def _check_color_answer(self, user_answer: str, correct_answer: str) -> Tuple[bool, str]:
        """检查颜色答案"""
        # 颜色标准化映射
        color_variations = {
            'red': ['red', 'crimson', 'scarlet'],
            'blue': ['blue', 'navy', 'azure', 'cyan'],
            'green': ['green', 'lime', 'emerald'],
            'yellow': ['yellow', 'gold', 'golden'],
            'orange': ['orange', 'amber'],
            'purple': ['purple', 'violet', 'magenta'],
            'black': ['black', 'dark'],
            'white': ['white', 'light'],
            'gray': ['gray', 'grey', 'silver'],
            'brown': ['brown', 'tan', 'bronze']
        }
        
        user_lower = user_answer.lower().strip()
        correct_lower = correct_answer.lower().strip()
        
        # 直接匹配
        if user_lower == correct_lower:
            return True, f"Correct! The answer is {correct_answer}"
        
        # 检查颜色变体
        for base_color, variations in color_variations.items():
            if correct_lower in variations:
                if any(var in user_lower for var in variations):
                    return True, f"Correct! The answer is {correct_answer}"
        
        return False, f"Incorrect. The correct answer is {correct_answer}"
    
    def _check_text_answer(self, user_answer: str, correct_answer: str) -> Tuple[bool, str]:
        """检查文本答案 - DVQA需要较严格的匹配"""
        # 清理和标准化
        user_clean = self._clean_text(user_answer).lower()
        correct_clean = self._clean_text(correct_answer).lower()
        
        # 1. 完全匹配（忽略大小写）
        if user_clean == correct_clean:
            return True, f"Correct! The answer is {correct_answer}"
        
        # 2. 对于单词答案，要求完全匹配
        if self.single_word_answer:
            # 提取用户答案中的单词
            user_words = user_clean.split()
            if correct_clean in user_words:
                return True, f"Correct! The answer is {correct_answer}"
            return False, f"Incorrect. The correct answer is {correct_answer}"
        
        # 3. 包含匹配（用户答案包含正确答案）
        if len(correct_clean) > 2 and correct_clean in user_clean:
            # 确保不是偶然包含
            if len(user_clean) <= len(correct_clean) * 2:
                return True, f"Correct! The answer is {correct_answer}"
        
        # 4. 对于短答案，检查编辑距离
        if len(correct_clean) <= 10:
            if self._check_edit_distance(user_clean, correct_clean, max_distance=1):
                return True, f"Correct! The answer is {correct_answer}"
        
        return False, f"Incorrect. The correct answer is {correct_answer}"
    
    def _clean_text(self, text: str) -> str:
        """清理文本 - 移除多余的空格和标点"""
        # 移除多余空格
        text = ' '.join(text.split())
        
        # 移除首尾的标点
        text = text.strip('.,!?;:')
        
        return text.strip()
    
    def _check_edit_distance(self, s1: str, s2: str, max_distance: int = 1) -> bool:
        """检查编辑距离（简单实现）"""
        # 对于短答案，允许1个字符的差异
        if abs(len(s1) - len(s2)) > max_distance:
            return False
        
        # 简单检查：字符差异
        if len(s1) == len(s2):
            diff_count = sum(1 for c1, c2 in zip(s1, s2) if c1 != c2)
            return diff_count <= max_distance
        
        return False
    
    def validate(
        self,
        chat_history: List[Dict],
        observation: Any,
        full_history: Optional[List[Any]] = None
    ) -> Tuple[float, bool, str, Dict[str, Any]]:
        """
        验证DVQA任务执行情况
        """
        # 调用父类验证
        reward, done, message, info = super().validate(
            chat_history, observation, full_history
        )
        
        # 添加DVQA特定的信息
        info["template_id"] = self.template_id
        info["question_type"] = self.question_type
        info["answer_type"] = self.answer_type
        info["chart_type"] = self.chart_type
        info["has_bbox"] = self.has_bbox
        info["is_numeric"] = self.is_numeric
        info["is_color_question"] = self.is_color_question
        info["is_comparison"] = self.is_comparison
        info["involves_counting"] = self.involves_counting
        info["single_word_answer"] = self.single_word_answer
        
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
            
            # 对于DVQA，通常期望简短答案
            if answer_length > 10:
                info["verbose_answer"] = True
            
            # 对于数值答案，提取数值
            if self.is_numeric:
                extracted_num = self._extract_number(provided_answer)
                if extracted_num is not None:
                    info["extracted_number"] = extracted_num
        
        return reward, done, message, info
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取DVQA特定的指标"""
        metrics = super().get_metrics()
        
        metrics.update({
            "template_id": self.template_id,
            "question_type": self.question_type,
            "answer_type": self.answer_type,
            "chart_type": self.chart_type,
            "has_bbox": self.has_bbox,
            "is_numeric": self.is_numeric,
            "is_color_question": self.is_color_question,
            "is_comparison": self.is_comparison,
            "involves_counting": self.involves_counting,
            "single_word_answer": self.single_word_answer,
            "chart_complexity": self._assess_chart_complexity(),
            "answer_complexity": self._assess_answer_complexity()
        })
        
        # 如果有边界框，添加面积信息
        if self.has_bbox:
            metrics["bbox_area"] = self._calculate_bbox_area()
        
        return metrics
    
    def _assess_chart_complexity(self) -> str:
        """评估图表复杂度"""
        complexity_factors = 0
        
        # 基于问题类型
        if self.question_type in ['comparison', 'counting']:
            complexity_factors += 1
        
        # 基于模板
        if self.template_id == 'reasoning':
            complexity_factors += 2
        elif self.template_id == 'structure':
            complexity_factors += 1
        
        # 基于答案类型
        if not self.single_word_answer:
            complexity_factors += 1
        
        # 返回复杂度等级
        if complexity_factors >= 3:
            return "high"
        elif complexity_factors >= 1:
            return "medium"
        else:
            return "low"
    
    def _assess_answer_complexity(self) -> str:
        """评估答案复杂度"""
        if self.answer_type in ['single_word', 'yes_no', 'numeric', 'color']:
            return "low"
        elif self.answer_type == 'phrase':
            return "medium"
        else:
            return "high"
    
    def get_chart_analysis_tips(self) -> Dict[str, List[str]]:
        """获取图表分析提示"""
        tips = {
            "general": [
                "识别图表类型和整体结构",
                "定位轴标签和图例",
                "理解数据的表示方法",
                "注意数值的单位和刻度"
            ],
            "bar": [
                "确定条形图是水平还是垂直",
                "通过条形的高度或长度读取数值",
                "注意是否有分组或堆叠",
                "检查轴上的刻度值"
            ],
            "line": [
                "跟踪线条的走势",
                "识别数据点的位置",
                "注意多条线的区分",
                "观察趋势和模式"
            ],
            "pie": [
                "识别扇形的大小",
                "检查百分比标签",
                "匹配颜色和图例",
                "记住总和为100%"
            ]
        }
        
        return tips
    
    def get_question_type_tips(self) -> Dict[str, str]:
        """获取问题类型提示"""
        tips = {
            "value_extraction": "定位特定数据点并读取精确数值",
            "comparison": "系统地比较所有相关数据点",
            "counting": "仔细计数所有相关元素",
            "color": "准确识别元素的颜色",
            "structure": "理解图表的组织和布局",
            "label_reading": "准确读取标签文本"
        }
        
        return tips.get(self.question_type, "仔细分析图表")