#!/usr/bin/env python3
"""
PlotQA Task implementation for VLM Gym
Handles plot/chart question answering with various question templates
"""

from typing import Tuple, Dict, Any, List, Optional
import re

from .vision_qa_task import VisionQATask


class PlotQATask(VisionQATask):
    """
    PlotQA 特定任务
    
    专门处理图表相关的视觉问答任务，支持6种问题模板：
    - structural: 图表结构问题（如"有多少条不同颜色的线？"）
    - data_retrieval: 数据提取（如"1993年的值是多少？"）
    - min_max: 最值问题（如"最大值是多少？"）
    - arithmetic: 算术计算（如"总和是多少？"）
    - comparison: 比较问题（如"有多少年的值大于X？"）
    - compound: 复合问题（结合多种操作）
    
    支持多种答案类型：
    - 数值（整数/浮点数）
    - Yes/No
    - 年份
    - 文本
    """
    
    # PlotQA的问题模板
    QUESTION_TEMPLATES = {
        'structural': 'Questions about chart structure and elements',
        'data_retrieval': 'Direct data extraction from charts',
        'min_max': 'Finding minimum or maximum values',
        'arithmetic': 'Calculations on chart data',
        'comparison': 'Comparing values in charts',
        'compound': 'Complex questions combining multiple operations'
    }
    
    # 图表类型
    CHART_TYPES = ['dot', 'line', 'bar', 'scatter', 'pie', 'area', 'box']
    
    @classmethod
    def get_task_id(cls) -> str:
        """获取任务类型ID"""
        return "vlm-gym.plotqa"
    
    def setup(self) -> Tuple[str, Dict[str, Any]]:
        """设置PlotQA特定的任务"""
        # 调用父类setup
        task_goal, task_info = super().setup()
        
        # 从metadata中提取PlotQA特定信息
        self.template = self.metadata.get('template', 'unknown')
        self.chart_type = self.metadata.get('chart_type', 'unknown')
        self.qid = self.metadata.get('qid', '')
        self.image_index = self.metadata.get('image_index', -1)
        self.answer_bbox = self.metadata.get('answer_bbox', [])
        
        # 分析答案类型
        self.answer_type = self._classify_answer_type()
        
        # 添加PlotQA特定的处理
        task_info["template"] = self.template
        task_info["chart_type"] = self.chart_type
        task_info["qid"] = self.qid
        task_info["answer_type"] = self.answer_type
        task_info["has_bbox"] = bool(self.answer_bbox)
        task_info["question_complexity"] = self._assess_question_complexity()
        task_info["requires_calculation"] = self.template in ['arithmetic', 'compound']
        task_info["requires_comparison"] = self.template == 'comparison'
        task_info["dataset"] = "plotqa"
        
        # 修改任务目标以包含PlotQA特定指导
        enhanced_goal = self._enhance_task_goal(task_goal)
        
        return enhanced_goal, task_info
    
    def _enhance_task_goal(self, base_goal: str) -> str:
        """增强任务目标描述，添加PlotQA特定的指导"""
        enhanced_parts = [base_goal]
        
        # 根据问题模板添加特定指导
        template_guidance = {
            'structural': "\nThis is a structural question. Count or identify the chart elements (lines, bars, etc.) carefully.",
            'data_retrieval': "\nThis is a data retrieval question. Read the exact value from the chart for the specified condition.",
            'min_max': "\nThis is a min/max question. Compare all values to find the extreme value or corresponding label.",
            'arithmetic': "\nThis is an arithmetic question. Extract all relevant values and perform the calculation step by step.",
            'comparison': "\nThis is a comparison question. Compare values systematically and count those meeting the criteria.",
            'compound': "\nThis is a compound question requiring multiple steps. Break it down and solve systematically."
        }
        
        if self.template in template_guidance:
            enhanced_parts.append(template_guidance[self.template])
        
        # 根据答案类型添加提示
        if self.answer_type == 'float':
            enhanced_parts.append("\nNote: The answer is a decimal number. Be precise in reading values from the chart.")
        elif self.answer_type == 'integer':
            enhanced_parts.append("\nNote: The answer should be a whole number.")
        elif self.answer_type == 'yes_no':
            enhanced_parts.append("\nNote: This is a yes/no question. Answer with 'Yes' or 'No'.")
        elif self.answer_type == 'year':
            enhanced_parts.append("\nNote: The answer is a year. Look for the year that satisfies the condition.")
        
        # 添加通用的图表分析提示
        enhanced_parts.append("\n\nWhen analyzing the chart:")
        enhanced_parts.append("\n- First identify the chart type and what it represents")
        enhanced_parts.append("\n- Check the axes labels and scales carefully")
        enhanced_parts.append("\n- For numerical values, be precise in reading from the chart")
        
        # 如果有边界框信息
        if self.answer_bbox:
            enhanced_parts.append("\n- The answer can be found in a specific region of the chart")
        
        return "\n".join(enhanced_parts)
    
    def _classify_answer_type(self) -> str:
        """分类答案类型"""
        if not self.answer:
            return 'unknown'
        
        answer_str = str(self.answer).strip()
        
        # Yes/No答案
        if answer_str.lower() in ['yes', 'no']:
            return 'yes_no'
        
        # 年份答案（4位数字，1900-2100）
        if re.match(r'^(19|20)\d{2}$', answer_str):
            return 'year'
        
        # 尝试解析为数字
        try:
            # 整数
            int_val = int(answer_str)
            # 检查是否真的是整数（没有小数部分）
            if str(int_val) == answer_str:
                return 'integer'
        except ValueError:
            pass
        
        try:
            # 浮点数
            float_val = float(answer_str)
            return 'float'
        except ValueError:
            pass
        
        # 其他都是文本
        return 'text'
    
    def _assess_question_complexity(self) -> str:
        """评估问题复杂度"""
        complexity_score = 0
        
        # 基于模板
        template_complexity = {
            'structural': 1,
            'data_retrieval': 1,
            'min_max': 2,
            'comparison': 2,
            'arithmetic': 3,
            'compound': 4
        }
        complexity_score += template_complexity.get(self.template, 2)
        
        # 基于问题长度
        if self.question:
            if len(self.question) > 80:
                complexity_score += 2
            elif len(self.question) > 50:
                complexity_score += 1
        
        # 基于答案类型
        if self.answer_type == 'float':
            complexity_score += 1  # 精确读数更难
        
        # 分类
        if complexity_score >= 5:
            return 'high'
        elif complexity_score >= 3:
            return 'medium'
        else:
            return 'low'
    
    def check_success(self, action: Any) -> Tuple[bool, str]:
        """
        检查PlotQA答案
        
        对于不同类型的答案采用不同的匹配策略
        """
        if not action:
            return False, "No answer provided"
        
        # 清理答案格式
        action_str = str(action).strip()
        correct_answer = str(self.answer).strip()
        
        # 1. Yes/No答案
        if self.answer_type == 'yes_no':
            if action_str.lower() == correct_answer.lower():
                return True, f"Correct! The answer is {correct_answer}"
            # 检查答案是否包含在句子中
            if correct_answer.lower() in action_str.lower():
                return True, f"Correct! You identified the answer: {correct_answer}"
            return False, f"Incorrect. The correct answer is {correct_answer}"
        
        # 2. 数值答案（整数、浮点数、年份）
        if self.answer_type in ['integer', 'float', 'year']:
            user_value = self._extract_number(action_str)
            correct_value = self._extract_number(correct_answer)
            
            if user_value is not None and correct_value is not None:
                # 对于整数和年份，要求精确匹配
                if self.answer_type in ['integer', 'year']:
                    if abs(user_value - correct_value) < 0.5:  # 允许四舍五入误差
                        return True, f"Correct! The answer is {int(correct_value)}"
                    else:
                        return False, f"Incorrect. Expected {int(correct_value)}, got {int(user_value)}"
                
                # 对于浮点数，允许一定误差
                else:  # float
                    # 计算相对误差
                    if correct_value == 0:
                        if abs(user_value - correct_value) <= 0.01:
                            return True, f"Correct! (exact match: {correct_value})"
                    else:
                        relative_error = abs(user_value - correct_value) / abs(correct_value)
                        if relative_error <= 0.01:  # 允许1%的误差
                            return True, f"Correct! The answer is {correct_value}"
                        elif relative_error <= 0.05:  # 5%误差给部分分
                            return True, f"Acceptable answer (within 5% of {correct_value})"
                    
                    return False, f"Incorrect. Expected {correct_value}, got {user_value}"
            
            # 如果无法提取数字，检查文本匹配
            if action_str == correct_answer:
                return True, f"Correct! The answer is {correct_answer}"
        
        # 3. 文本答案（默认）
        # 精确匹配
        if action_str.lower() == correct_answer.lower():
            return True, f"Correct! The answer is {correct_answer}"
        
        # 检查答案是否包含在响应中
        if self._contains_answer(action_str, correct_answer):
            return True, f"Correct! You identified the answer: {correct_answer}"
        
        return False, f"Incorrect. The correct answer is {correct_answer}"
    
    def _extract_number(self, text: str) -> Optional[float]:
        """从文本中提取数字"""
        if not text:
            return None
        
        # 清理文本
        text = text.lower().strip()
        
        # 移除常见的干扰字符和词
        text = text.replace(',', '').replace('$', '').replace('approximately', '').replace('about', '')
        
        # 特殊处理百分比
        is_percentage = '%' in text or 'percent' in text
        text = text.replace('%', '').replace('percent', '')
        
        # 查找数字模式（支持负数和小数）
        number_pattern = r'-?\d+\.?\d*'
        matches = re.findall(number_pattern, text)
        
        if matches:
            try:
                # 优先返回最后一个数字（通常是答案）
                number = float(matches[-1])
                if is_percentage and number > 1:
                    number = number / 100
                return number
            except ValueError:
                pass
        
        return None
    
    def _contains_answer(self, response: str, answer: str) -> bool:
        """检查响应是否包含答案"""
        response_lower = response.lower()
        answer_lower = answer.lower()
        
        # 直接包含
        if answer_lower in response_lower:
            return True
        
        # 对于数字答案，检查不同格式
        if self.answer_type in ['integer', 'float', 'year']:
            answer_num = self._extract_number(answer)
            if answer_num is not None:
                # 检查不同的数字格式
                variations = [
                    str(int(answer_num)),  # 整数形式
                    f"{answer_num:.1f}",   # 一位小数
                    f"{answer_num:.2f}",   # 两位小数
                    f"{answer_num:.4f}",   # 四位小数
                ]
                
                for var in variations:
                    if var in response:
                        return True
        
        return False
    
    def validate(
        self,
        chat_history: List[Dict],
        observation: Any,
        full_history: Optional[List[Any]] = None
    ) -> Tuple[float, bool, str, Dict[str, Any]]:
        """
        验证PlotQA任务执行情况
        """
        # 调用父类验证
        reward, done, message, info = super().validate(
            chat_history, observation, full_history
        )
        
        # 添加PlotQA特定的信息
        info["template"] = self.template
        info["chart_type"] = self.chart_type
        info["qid"] = self.qid
        info["answer_type"] = self.answer_type
        info["question_complexity"] = self._assess_question_complexity()
        info["has_bbox"] = bool(self.answer_bbox)
        
        # 分析答案质量
        if info.get("answer_provided"):
            provided_answer = str(info["answer_provided"])
            
            # 对于数值答案，提取数值
            if self.answer_type in ['integer', 'float', 'year']:
                extracted_value = self._extract_number(provided_answer)
                if extracted_value is not None:
                    info["extracted_numerical_value"] = extracted_value
                    correct_value = self._extract_number(str(self.answer))
                    if correct_value is not None:
                        if correct_value != 0:
                            info["relative_error"] = abs(extracted_value - correct_value) / abs(correct_value)
                        else:
                            info["absolute_error"] = abs(extracted_value - correct_value)
                else:
                    info["numerical_extraction_failed"] = True
            
            # 检查答案长度（过长可能表示不确定）
            if len(provided_answer) > 200:
                info["verbose_answer"] = True
        
        return reward, done, message, info
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取PlotQA特定的指标"""
        metrics = super().get_metrics()
        
        metrics.update({
            "template": self.template,
            "chart_type": self.chart_type,
            "qid": self.qid,
            "answer_type": self.answer_type,
            "question_complexity": self._assess_question_complexity(),
            "has_bbox": bool(self.answer_bbox),
            "image_index": self.image_index,
            "is_numerical": self.answer_type in ['integer', 'float', 'year'],
            "is_calculation": self.template in ['arithmetic', 'compound'],
            "is_comparison": self.template == 'comparison',
            "is_structural": self.template == 'structural',
            "requires_precise_reading": self.answer_type == 'float'
        })
        
        return metrics
    
    def get_template_description(self) -> str:
        """获取问题模板的描述"""
        return self.QUESTION_TEMPLATES.get(self.template, "Unknown question type")
    
    def get_difficulty_factors(self) -> List[str]:
        """获取影响难度的因素"""
        factors = []
        
        # 模板相关
        if self.template in ['arithmetic', 'compound']:
            factors.append("requires_calculation")
        if self.template == 'comparison':
            factors.append("requires_systematic_comparison")
        
        # 答案类型相关
        if self.answer_type == 'float':
            factors.append("precise_value_reading")
        if self.answer_type == 'year':
            factors.append("temporal_reasoning")
        
        # 问题长度
        if self.question and len(self.question) > 80:
            factors.append("complex_question")
        
        # 图表类型
        if self.chart_type in ['scatter', 'box']:
            factors.append("complex_chart_type")
        
        return factors