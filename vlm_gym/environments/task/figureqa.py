"""
FigureQA Task implementation for VLM Gym
Handles binary visual question answering tasks on scientific figures
"""

from typing import Tuple, Dict, Any, List, Optional
import re

from .vision_qa_task import VisionQATask


class FigureQATask(VisionQATask):
    """
    FigureQA 特定任务
    
    专门处理科学图表的二元问答任务，包括：
    - 最值识别（minimum/maximum）
    - 数值比较（greater than/less than）
    - 中位数判断（high median/low median）
    
    所有问题都是Yes/No二元选择
    """
    
    # FigureQA的问题类型
    QUESTION_TYPES = {
        'minimum': 'Finding the minimum value',
        'maximum': 'Finding the maximum value',
        'greater_than': 'Comparing two values (greater than)',
        'less_than': 'Comparing two values (less than)',
        'high_median': 'Identifying the high median',
        'low_median': 'Identifying the low median'
    }
    
    # 图表类型（基于FigureQA数据集）
    CHART_TYPES = [
        'bar_chart',
        'line_plot', 
        'scatter_plot',
        'pie_chart',
        'area_chart'
    ]
    
    @classmethod
    def get_task_id(cls) -> str:
        """获取任务类型ID"""
        return "vlm-gym.figureqa"
    
    def setup(self) -> Tuple[str, Dict[str, Any]]:
        """设置FigureQA特定的任务"""
        # 调用父类setup
        task_goal, task_info = super().setup()
        
        # 从metadata中提取FigureQA特定信息
        self.question_id = self.metadata.get('question_id', -1)
        self.image_index = self.metadata.get('image_index', -1)
        self.color1_name = self.metadata.get('color1_name', '')
        self.color2_name = self.metadata.get('color2_name', '')
        self.color1_rgb = self.metadata.get('color1_rgb', [])
        self.color2_rgb = self.metadata.get('color2_rgb', [])
        
        # 确保答案是Yes/No格式
        self.yes_no_answer = self.task_data.get('yes_no_answer', '')
        
        # 分类问题类型
        question_type = self._classify_question_type()
        
        # 添加FigureQA特定的处理
        task_info["question_type"] = question_type
        task_info["question_type_desc"] = self.QUESTION_TYPES.get(question_type, "Unknown")
        task_info["involves_colors"] = self._involves_colors()
        task_info["chart_type"] = self._detect_chart_type()
        task_info["dataset"] = "figureqa"
        task_info["binary_task"] = True
        task_info["expected_answer_format"] = "Yes/No"
        
        # 修改任务目标以包含FigureQA特定指导
        enhanced_goal = self._enhance_task_goal(task_goal, question_type)
        
        return enhanced_goal, task_info
    
    def _enhance_task_goal(self, base_goal: str, question_type: str) -> str:
        """增强任务目标描述，添加FigureQA特定的指导"""
        enhanced_parts = [base_goal]
        
        # 添加二元答案说明
        enhanced_parts.append("\n**Important**: This is a Yes/No question. Please answer with either 'Yes' or 'No'.")
        
        # 根据问题类型添加具体指导
        if question_type == 'minimum':
            enhanced_parts.append("\nTo find the minimum:")
            enhanced_parts.append("1. Identify all data points in the figure")
            enhanced_parts.append("2. Compare their values")
            enhanced_parts.append("3. Check if the mentioned item/color has the smallest value")
            
        elif question_type == 'maximum':
            enhanced_parts.append("\nTo find the maximum:")
            enhanced_parts.append("1. Identify all data points in the figure")
            enhanced_parts.append("2. Compare their values")
            enhanced_parts.append("3. Check if the mentioned item/color has the largest value")
            
        elif question_type in ['greater_than', 'less_than']:
            enhanced_parts.append("\nTo compare values:")
            enhanced_parts.append("1. Locate the two items/colors being compared")
            enhanced_parts.append("2. Read their exact values from the figure")
            enhanced_parts.append("3. Perform the comparison as asked")
            
        elif question_type in ['high_median', 'low_median']:
            enhanced_parts.append("\nTo find the median:")
            enhanced_parts.append("1. Identify all data values in the figure")
            enhanced_parts.append("2. Sort them in order")
            enhanced_parts.append("3. Find the median value(s)")
            enhanced_parts.append("4. Check if the mentioned item matches the median")
        
        # 如果涉及颜色，添加颜色信息
        if self._involves_colors():
            enhanced_parts.append(f"\nColor information:")
            if self.color1_name and self.color1_name != '--None--':
                enhanced_parts.append(f"- {self.color1_name}: RGB{self.color1_rgb}")
            if self.color2_name and self.color2_name != '--None--':
                enhanced_parts.append(f"- {self.color2_name}: RGB{self.color2_rgb}")
        
        return "\n".join(enhanced_parts)
    
    def _classify_question_type(self) -> str:
        """分类FigureQA问题类型"""
        if not self.question:
            return 'unknown'
        
        question_lower = self.question.lower()
        
        # 直接匹配问题类型
        if 'minimum' in question_lower:
            return 'minimum'
        elif 'maximum' in question_lower:
            return 'maximum'
        elif 'greater than' in question_lower:
            return 'greater_than'
        elif 'less than' in question_lower:
            return 'less_than'
        elif 'high median' in question_lower:
            return 'high_median'
        elif 'low median' in question_lower:
            return 'low_median'
        
        return 'unknown'
    
    def _involves_colors(self) -> bool:
        """判断问题是否涉及颜色"""
        has_color1 = self.color1_name and self.color1_name != '--None--'
        has_color2 = self.color2_name and self.color2_name != '--None--'
        return has_color1 or has_color2
    
    def _detect_chart_type(self) -> str:
        """检测图表类型（基于问题内容推断）"""
        # FigureQA数据集包含多种图表类型
        # 由于无法直接从数据中获取，这里基于经验推断
        question_lower = self.question.lower() if self.question else ""
        
        # 基于问题模式推断可能的图表类型
        if any(word in question_lower for word in ['bar', 'column']):
            return 'bar_chart'
        elif any(word in question_lower for word in ['line', 'trend', 'curve']):
            return 'line_plot'
        elif any(word in question_lower for word in ['scatter', 'point', 'dot']):
            return 'scatter_plot'
        elif any(word in question_lower for word in ['pie', 'slice', 'portion']):
            return 'pie_chart'
        elif any(word in question_lower for word in ['area', 'region']):
            return 'area_chart'
        
        # 默认假设是常见的类型
        return 'unknown'
    
    def check_success(self, action: Any) -> Tuple[bool, str]:
        """
        检查FigureQA答案 - 只接受Yes/No答案
        """
        if not action:
            return False, "No answer provided"
        
        # 清理答案格式
        action_str = str(action).strip()
        
        # 获取正确答案
        correct_answer = self.yes_no_answer.lower()  # 应该是 'yes' 或 'no'
        
        # 提取用户的Yes/No答案
        user_answer = self._extract_yes_no(action_str)
        
        if user_answer is None:
            return False, "Please answer with 'Yes' or 'No'. Your response was not recognized as a valid answer."
        
        # 检查答案是否正确
        if user_answer == correct_answer:
            # 提供详细的反馈
            question_type = self._classify_question_type()
            if question_type in ['minimum', 'maximum']:
                feedback = f"Correct! You correctly identified that the answer to the {question_type} question is {correct_answer.capitalize()}."
            elif question_type in ['greater_than', 'less_than']:
                feedback = f"Correct! The comparison is {correct_answer.capitalize()}."
            elif question_type in ['high_median', 'low_median']:
                feedback = f"Correct! The median identification is {correct_answer.capitalize()}."
            else:
                feedback = f"Correct! The answer is {correct_answer.capitalize()}."
            return True, feedback
        else:
            return False, f"Incorrect. The correct answer is '{correct_answer.capitalize()}', but you answered '{user_answer.capitalize()}'."
    
    def _extract_yes_no(self, text: str) -> Optional[str]:
        """从文本中提取Yes/No答案"""
        text_lower = text.lower().strip()
        
        # 直接匹配
        if text_lower in ['yes', 'no']:
            return text_lower
        
        # 匹配包含yes/no的短语
        if text_lower in ['yes.', 'no.', 'yes!', 'no!']:
            return text_lower.rstrip('.!')
        
        # 🔴 添加检查：如果同时包含yes和no，返回None
        if 'yes' in text_lower and 'no' in text_lower:
            return None
        
        # 在句子中查找
        # 匹配独立的yes/no词
        yes_pattern = r'\b(yes)\b'
        no_pattern = r'\b(no)\b'
        
        yes_match = re.search(yes_pattern, text_lower)
        no_match = re.search(no_pattern, text_lower)
        
        # 如果只找到一个，返回它
        if yes_match and not no_match:
            return 'yes'
        elif no_match and not yes_match:
            return 'no'
        # 🔴 移除了同时匹配的处理，因为上面已经处理了
        
        # 检查其他肯定/否定表达
        affirmative_phrases = ['correct', 'true', 'right', 'affirmative', 'indeed', 'certainly']
        negative_phrases = ['incorrect', 'false', 'wrong', 'negative', 'not', "doesn't", "isn't"]
        
        for phrase in affirmative_phrases:
            if phrase in text_lower:
                return 'yes'
        
        for phrase in negative_phrases:
            if phrase in text_lower:
                return 'no'
        
        return None
    
    def validate(
        self,
        chat_history: List[Dict],
        observation: Any,
        full_history: Optional[List[Any]] = None
    ) -> Tuple[float, bool, str, Dict[str, Any]]:
        """
        验证FigureQA任务执行情况
        """
        # 调用父类验证
        reward, done, message, info = super().validate(
            chat_history, observation, full_history
        )
        
        # 添加FigureQA特定的信息
        info["question_type"] = self._classify_question_type()
        info["involves_colors"] = self._involves_colors()
        info["chart_type"] = self._detect_chart_type()
        info["expected_answer"] = self.yes_no_answer
        info["binary_task"] = True
        
        # 分析答案质量
        if info.get("answer_provided"):
            provided_answer = str(info["answer_provided"])
            extracted_answer = self._extract_yes_no(provided_answer)
            
            info["valid_yes_no_format"] = extracted_answer is not None
            if extracted_answer:
                info["extracted_answer"] = extracted_answer
            else:
                info["format_issue"] = "Answer not in Yes/No format"
        
        # 添加颜色信息
        if self._involves_colors():
            info["color_info"] = {
                "color1": {"name": self.color1_name, "rgb": self.color1_rgb},
                "color2": {"name": self.color2_name, "rgb": self.color2_rgb}
            }
        
        return reward, done, message, info
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取FigureQA特定的指标"""
        metrics = super().get_metrics()
        
        question_type = self._classify_question_type()
        
        metrics.update({
            "question_type": question_type,
            "question_type_category": self._get_question_category(question_type),
            "involves_colors": self._involves_colors(),
            "num_colors_involved": self._count_colors(),
            "chart_type": self._detect_chart_type(),
            "is_comparison": question_type in ['greater_than', 'less_than'],
            "is_extreme_value": question_type in ['minimum', 'maximum'],
            "is_median": question_type in ['high_median', 'low_median'],
            "expected_answer": self.yes_no_answer,
            "answer_distribution": self._get_answer_distribution()
        })
        
        return metrics
    
    def _get_question_category(self, question_type: str) -> str:
        """获取问题的大类别"""
        if question_type in ['minimum', 'maximum']:
            return 'extreme_value'
        elif question_type in ['greater_than', 'less_than']:
            return 'comparison'
        elif question_type in ['high_median', 'low_median']:
            return 'median'
        else:
            return 'other'
    
    def _count_colors(self) -> int:
        """计算涉及的颜色数量"""
        count = 0
        if self.color1_name and self.color1_name != '--None--':
            count += 1
        if self.color2_name and self.color2_name != '--None--':
            count += 1
        return count
    
    def _get_answer_distribution(self) -> str:
        """获取答案分布（用于数据集平衡性分析）"""
        # 这里只返回当前问题的答案
        # 实际使用时可以聚合统计整个数据集的分布
        return self.yes_no_answer.lower()
    
    def get_hint(self) -> str:
        """获取问题提示（用于辅助）"""
        question_type = self._classify_question_type()
        
        hints = {
            'minimum': "Look for the smallest value among all data points in the figure.",
            'maximum': "Look for the largest value among all data points in the figure.",
            'greater_than': "Compare the two mentioned values directly. Is the first one larger?",
            'less_than': "Compare the two mentioned values directly. Is the first one smaller?",
            'high_median': "Find the median of all values. For even number of values, high median is the larger of the two middle values.",
            'low_median': "Find the median of all values. For even number of values, low median is the smaller of the two middle values."
        }
        
        base_hint = hints.get(question_type, "Carefully analyze the figure and answer with Yes or No.")
        
        # 如果涉及颜色，添加颜色提示
        if self._involves_colors():
            base_hint += "\nPay attention to the colors mentioned in the question."
        
        return base_hint