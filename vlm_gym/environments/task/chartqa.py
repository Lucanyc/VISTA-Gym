# vlm_gym/environments/task/chartqa_task.py

from typing import Tuple, Dict, Any, List, Optional
import re

from .vision_qa_task import VisionQATask


class ChartQATask(VisionQATask):
    """
    ChartQA 特定任务
    
    专门处理图表相关的视觉问答任务，包括：
    - 数据提取
    - 趋势分析
    - 数值计算
    - 图表比较
    """
    
    @classmethod
    def get_task_id(cls) -> str:
        """获取任务类型ID"""
        return "vlm-gym.chart-qa"
    
    def setup(self) -> Tuple[str, Dict[str, Any]]:
        """设置ChartQA特定的任务"""
        # 调用父类setup
        task_goal, task_info = super().setup()
        
        # 添加ChartQA特定的处理
        task_info["chart_type"] = self._detect_chart_type()
        task_info["requires_calculation"] = self._requires_calculation()
        task_info["data_extraction_needed"] = self._needs_data_extraction()
        task_info["dataset"] = "chartqa"
        
        # 修改任务目标以包含图表特定指导
        enhanced_goal = task_goal
        
        if task_info["requires_calculation"]:
            enhanced_goal += "\n\nNote: This question may require numerical calculations based on the chart data."
        
        if task_info["data_extraction_needed"]:
            enhanced_goal += "\nPlease carefully extract the relevant data points from the chart before answering."
        
        # 添加ChartQA特定的提示
        enhanced_goal += "\n\nWhen analyzing the chart, please:"
        enhanced_goal += "\n- Identify the chart type and axes"
        enhanced_goal += "\n- Read values accurately from the chart"
        enhanced_goal += "\n- Show your reasoning if calculations are needed"
        
        return enhanced_goal, task_info
    
    def _detect_chart_type(self) -> str:
        """检测图表类型"""
        question_lower = self.question.lower() if self.question else ""
        
        # 基于问题内容推断图表类型
        chart_keywords = {
            "bar_chart": ["bar", "column", "bars"],
            "line_chart": ["line", "trend", "over time", "timeline"],
            "pie_chart": ["pie", "percentage", "portion", "share", "distribution"],
            "scatter_plot": ["scatter", "correlation", "relationship"],
            "area_chart": ["area", "cumulative"],
            "histogram": ["histogram", "frequency", "bins"]
        }
        
        for chart_type, keywords in chart_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                return chart_type
        
        # 基于问题模式推断
        if "compare" in question_lower or "difference" in question_lower:
            return "comparative_chart"
        elif "total" in question_lower or "sum" in question_lower:
            return "aggregated_chart"
        
        return "unknown"
    
    def _requires_calculation(self) -> bool:
        """判断是否需要计算"""
        if not self.question:
            return False
        
        calculation_keywords = [
            "calculate", "compute", "sum", "total", "average", "mean",
            "difference", "ratio", "percentage", "increase", "decrease",
            "compare", "how many", "how much", "what is the",
            "growth", "decline", "change", "rate"
        ]
        
        question_lower = self.question.lower()
        return any(keyword in question_lower for keyword in calculation_keywords)
    
    def _needs_data_extraction(self) -> bool:
        """判断是否需要数据提取"""
        if not self.question:
            return False
        
        extraction_keywords = [
            "value", "data", "point", "highest", "lowest", "maximum",
            "minimum", "specific", "exact", "what is", "which", "find",
            "identify", "locate", "read"
        ]
        
        question_lower = self.question.lower()
        return any(keyword in question_lower for keyword in extraction_keywords)
    
    def check_success(self, action: Any) -> Tuple[bool, str]:
        """
        检查ChartQA答案
        
        对于数值答案，允许一定的误差范围
        """
        if not action:
            return False, "No answer provided"
        
        # 清理答案格式
        action = str(action).strip()
        
        # 首先尝试父类的检查
        success, feedback = super().check_success(action)
        
        # 如果父类检查失败，尝试ChartQA特定的检查
        if not success and self.answer:
            # 检查数值答案（允许误差）
            user_value = self._extract_number(action)
            correct_value = self._extract_number(str(self.answer))
            
            if user_value is not None and correct_value is not None:
                # 计算相对误差
                if correct_value == 0:
                    # 绝对误差检查
                    if abs(user_value - correct_value) <= 0.1:
                        return True, f"Correct! (exact match for zero value)"
                else:
                    relative_error = abs(user_value - correct_value) / abs(correct_value)
                    if relative_error <= 0.05:  # 允许5%的误差
                        return True, f"Correct! (within acceptable range: {correct_value:.2f} ± 5%)"
                    elif relative_error <= 0.1:  # 10%误差给部分分
                        return True, f"Acceptable answer (within 10% of {correct_value:.2f})"
                
                return False, f"Incorrect. Expected approximately {correct_value:.2f}, got {user_value:.2f}"
            
            # 检查单位转换
            if self._check_unit_conversion(action, str(self.answer)):
                return True, "Correct! (with unit conversion)"
        
        return success, feedback
    
    def _extract_number(self, text: str) -> Optional[float]:
        """从文本中提取数字"""
        if not text:
            return None
        
        # 清理文本
        text = text.lower()
        
        # 移除常见的干扰字符
        text = text.replace(',', '').replace('%', '').replace('$', '')
        
        # 处理百分比
        is_percentage = '%' in text or 'percent' in text
        
        # 处理单位
        unit_multipliers = {
            'k': 1000, 'thousand': 1000, 'thousands': 1000,
            'm': 1000000, 'million': 1000000, 'millions': 1000000,
            'b': 1000000000, 'billion': 1000000000, 'billions': 1000000000
        }
        
        multiplier = 1
        for unit, value in unit_multipliers.items():
            if unit in text:
                multiplier = value
                text = text.replace(unit, '')
        
        # 查找数字模式
        number_pattern = r'-?\d+\.?\d*'
        matches = re.findall(number_pattern, text)
        
        if matches:
            try:
                # 返回找到的第一个数字
                number = float(matches[0])
                if is_percentage and number > 1:
                    number = number / 100
                return number * multiplier
            except ValueError:
                pass
        
        return None
    
    def _check_unit_conversion(self, user_answer: str, correct_answer: str) -> bool:
        """检查是否是单位转换导致的差异"""
        user_value = self._extract_number(user_answer)
        correct_value = self._extract_number(correct_answer)
        
        if user_value is None or correct_value is None:
            return False
        
        # 检查常见的单位转换比例
        common_ratios = [1000, 100, 12, 60, 24, 365]  # K/M, %, 月/年, 分/时, 时/天, 天/年
        
        for ratio in common_ratios:
            if abs(user_value * ratio - correct_value) < 0.01 * correct_value:
                return True
            if abs(user_value / ratio - correct_value) < 0.01 * correct_value:
                return True
        
        return False
    
    def validate(
        self,
        chat_history: List[Dict],
        observation: Any,
        full_history: Optional[List[Any]] = None
    ) -> Tuple[float, bool, str, Dict[str, Any]]:
        """
        验证ChartQA任务执行情况
        """
        # 调用父类验证
        reward, done, message, info = super().validate(
            chat_history, observation, full_history
        )
        
        # 添加ChartQA特定的信息
        info["chart_type"] = self._detect_chart_type()
        info["required_calculation"] = self._requires_calculation()
        info["data_extraction"] = self._needs_data_extraction()
        
        # 如果是数值答案，添加提取的数值信息
        if info.get("answer_provided"):
            extracted_value = self._extract_number(str(info["answer_provided"]))
            if extracted_value is not None:
                info["extracted_numerical_value"] = extracted_value
        
        return reward, done, message, info
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取ChartQA特定的指标"""
        metrics = super().get_metrics()
        
        metrics.update({
            "chart_type": self._detect_chart_type(),
            "requires_calculation": self._requires_calculation(),
            "needs_data_extraction": self._needs_data_extraction(),
            "is_numerical_answer": self._extract_number(str(self.answer)) is not None if self.answer else False,
            "question_complexity": self._assess_complexity()
        })
        
        return metrics
    
    def _assess_complexity(self) -> str:
        """评估问题复杂度"""
        if not self.question:
            return "unknown"
        
        question_lower = self.question.lower()
        
        # 复杂问题特征
        complex_features = [
            "compare", "calculate", "trend", "correlation",
            "percentage change", "growth rate", "difference between"
        ]
        
        # 简单问题特征
        simple_features = [
            "what is", "which", "highest", "lowest", "maximum", "minimum"
        ]
        
        complex_count = sum(1 for f in complex_features if f in question_lower)
        simple_count = sum(1 for f in simple_features if f in question_lower)
        
        if complex_count >= 2:
            return "high"
        elif complex_count >= 1 or (self._requires_calculation() and self._needs_data_extraction()):
            return "medium"
        elif simple_count >= 1:
            return "low"
        else:
            return "medium"