"""
TabMWP Task - Tabular Math Word Problems
用于处理表格数学文字问题的视觉问答任务
"""

from typing import Tuple, Dict, Any, List, Optional
import re
import math
import logging
from fractions import Fraction

from vlm_gym.environments.task.vision_qa_task import VisionQATask

logger = logging.getLogger(__name__)


class TabMWPTask(VisionQATask):
    """
    TabMWP (Tabular Math Word Problems) 特定任务
    
    专门处理表格数学文字问题，包括：
    - 表格数据分析
    - 数学计算和推理
    - 统计分析
    - 比较和排序
    - 分数和百分比计算
    - 频率表分析
    - 茎叶图解读
    """
    
    @classmethod
    def get_task_id(cls) -> str:
        """获取任务类型ID"""
        return "vlm-gym.tabmwp"
    
    def setup(self) -> Tuple[str, Dict[str, Any]]:
        """设置TabMWP特定的任务"""
        # 调用父类setup
        task_goal, task_info = super().setup()
        
        # 添加TabMWP特定的处理
        task_info["problem_type"] = self._detect_problem_type()
        task_info["calculation_type"] = self._detect_calculation_type()
        task_info["table_type"] = self._detect_table_type()
        task_info["requires_multi_step"] = self._requires_multi_step()
        task_info["statistical_concept"] = self._detect_statistical_concept()
        task_info["dataset"] = "tabmwp"
        
        # 从metadata中提取信息
        if hasattr(self, 'metadata') and self.metadata:
            task_info["grade_level"] = self.metadata.get("grade_level", 0)
            task_info["answer_format"] = self.metadata.get("answer_format", "text")
            task_info["table_dimensions"] = self.metadata.get("table_dimensions", {})
            task_info["problem_features"] = self.metadata.get("problem_features", {})
            task_info["has_solution"] = bool(self.metadata.get("solution_explanation"))
        
        # 修改任务目标以包含TabMWP特定指导
        enhanced_goal = task_goal
        
        # 添加表格分析指导
        enhanced_goal += "\n\n**Table Analysis Guidelines:**"
        enhanced_goal += "\n- Carefully examine the table structure and data organization"
        enhanced_goal += "\n- Identify column headers, row labels, and data patterns"
        enhanced_goal += "\n- Extract relevant numerical values from the table"
        
        # 根据问题类型添加特定指导
        problem_type = task_info["problem_type"]
        if problem_type == "counting":
            enhanced_goal += "\n\n**Counting Task:** Count the number of items that meet the specified criteria."
        elif problem_type == "comparison":
            enhanced_goal += "\n\n**Comparison Task:** Compare values and identify the highest, lowest, or differences."
        elif problem_type == "calculation":
            enhanced_goal += "\n\n**Calculation Task:** Perform arithmetic operations on the table data."
        elif problem_type == "statistics":
            enhanced_goal += "\n\n**Statistics Task:** Calculate statistical measures like mean, median, mode, or range."
        elif problem_type == "fraction":
            enhanced_goal += "\n\n**Fraction Task:** Express answers as fractions and simplify when needed."
        
        # 根据表格类型添加指导
        table_type = task_info["table_type"]
        if table_type == "stem_and_leaf":
            enhanced_goal += "\n\n**Stem-and-Leaf Plot:** Each stem represents the tens digit, each leaf represents the units digit."
        elif table_type == "frequency":
            enhanced_goal += "\n\n**Frequency Table:** Count how many times each value appears."
        elif table_type == "data_table":
            enhanced_goal += "\n\n**Data Table:** Extract and analyze the numerical data systematically."
        
        # 添加年级水平相关的提示
        grade = task_info.get("grade_level", 0)
        if grade <= 4:
            enhanced_goal += "\n\n**Elementary Level:** Focus on basic counting, addition, and simple comparisons."
        elif grade <= 6:
            enhanced_goal += "\n\n**Middle Grade:** May involve multiplication, division, and basic statistics."
        else:
            enhanced_goal += "\n\n**Advanced Level:** May require complex calculations and statistical reasoning."
        
        # 如果有选择题，添加提示
        if hasattr(self, 'choices') and self.choices:
            enhanced_goal += f"\n\nChoose from: {', '.join(self.choices)}"
        
        return enhanced_goal, task_info
    
    def _detect_problem_type(self) -> str:
        """检测问题类型"""
        if not self.question:
            return "unknown"
        
        question_lower = self.question.lower()
        
        # 计数问题
        if any(phrase in question_lower for phrase in ["how many", "count", "number of"]):
            return "counting"
        
        # 比较问题
        if any(phrase in question_lower for phrase in ["highest", "lowest", "greatest", "least", "most", "fewest", "which"]):
            return "comparison"
        
        # 分数问题
        if any(phrase in question_lower for phrase in ["fraction", "what fraction", "simplify"]):
            return "fraction"
        
        # 统计问题
        if any(phrase in question_lower for phrase in ["average", "mean", "median", "mode", "range"]):
            return "statistics"
        
        # 计算问题
        if any(phrase in question_lower for phrase in ["total", "sum", "add", "difference", "calculate"]):
            return "calculation"
        
        # 函数问题
        if any(phrase in question_lower for phrase in ["function", "linear", "nonlinear"]):
            return "function_analysis"
        
        return "general"
    
    def _detect_calculation_type(self) -> str:
        """检测计算类型"""
        if not self.question:
            return "none"
        
        question_lower = self.question.lower()
        
        if any(word in question_lower for word in ["add", "sum", "total", "altogether"]):
            return "addition"
        elif any(word in question_lower for word in ["subtract", "difference", "minus", "less"]):
            return "subtraction"
        elif any(word in question_lower for word in ["multiply", "times", "product"]):
            return "multiplication"
        elif any(word in question_lower for word in ["divide", "divided", "quotient", "per"]):
            return "division"
        elif any(word in question_lower for word in ["percentage", "percent", "%"]):
            return "percentage"
        elif any(word in question_lower for word in ["fraction", "ratio"]):
            return "fraction"
        
        return "none"
    
    def _detect_table_type(self) -> str:
        """检测表格类型"""
        # 从metadata获取表格信息，确保处理None值
        metadata = self.metadata or {}
        table_text = (metadata.get("table_text") or "").lower()
        table_title = (metadata.get("table_title") or "").lower()
        
        # 茎叶图
        if "stem" in table_text and "leaf" in table_text:
            return "stem_and_leaf"
        elif "stem" in table_title and "leaf" in table_title:
            return "stem_and_leaf"
        
        # 频率表
        if "frequency" in table_text or "frequency" in table_title:
            return "frequency"
        
        # 函数表
        if "x" in table_text and "y" in table_text:
            return "function_table"
        
        # 价格表
        if "$" in table_text or "price" in table_title or "cost" in table_title:
            return "price_table"
        
        # 一般数据表
        return "data_table"
    
    def _requires_multi_step(self) -> bool:
        """判断是否需要多步骤推理"""
        if not self.question:
            return False
        
        # 检查解题步骤说明
        solution = (self.metadata.get("solution_explanation") or "") if self.metadata else ""
        if solution:
            # 如果解题步骤包含多个步骤标识
            step_indicators = ["step", "first", "then", "next", "finally", "add:", "find", "now"]
            step_count = sum(1 for indicator in step_indicators if indicator in solution.lower())
            return step_count >= 2
        
        # 基于问题复杂度判断
        question_lower = self.question.lower()
        multi_step_keywords = ["then", "and", "total", "altogether", "in all", "difference between"]
        return any(keyword in question_lower for keyword in multi_step_keywords)
    
    def _detect_statistical_concept(self) -> str:
        """检测统计概念"""
        if not self.question:
            return "none"
        
        question_lower = self.question.lower()
        
        if "average" in question_lower or "mean" in question_lower:
            return "mean"
        elif "median" in question_lower:
            return "median"
        elif "mode" in question_lower:
            return "mode"
        elif "range" in question_lower:
            return "range"
        elif "frequency" in question_lower:
            return "frequency"
        elif "distribution" in question_lower:
            return "distribution"
        
        return "none"
    
    def check_success(self, action: Any) -> Tuple[bool, str]:
        """
        检查TabMWP答案
        
        支持多种答案格式：整数、小数、分数、文本
        """
        if not action:
            return False, "No answer provided"
        
        # 清理答案格式
        action = str(action).strip()
        
        # 首先尝试父类的检查
        success, feedback = super().check_success(action)
        
        # 如果父类检查失败，尝试TabMWP特定的检查
        if not success and self.answer:
            correct_answer = str(self.answer).strip()
            
            # 检查分数答案
            if self._is_fraction_answer(correct_answer):
                success, feedback = self._check_fraction_answer(action, correct_answer)
                if success:
                    return success, feedback
            
            # 检查数值答案（允许误差）
            user_value = self._extract_number(action)
            correct_value = self._extract_number(correct_answer)
            
            if user_value is not None and correct_value is not None:
                # 对于整数答案，要求精确匹配
                if self.metadata and self.metadata.get("answer_format") == "integer":
                    if abs(user_value - correct_value) < 0.001:
                        return True, "Correct!"
                    else:
                        return False, f"Incorrect. Expected {int(correct_value)}, got {user_value}"
                
                # 对于小数答案，允许小的误差
                elif self.metadata and self.metadata.get("answer_format") == "decimal":
                    if abs(user_value - correct_value) <= 0.01:
                        return True, "Correct!"
                    else:
                        return False, f"Incorrect. Expected {correct_value:.2f}, got {user_value:.2f}"
                
                # 一般数值比较
                else:
                    relative_error = abs(user_value - correct_value) / max(abs(correct_value), 1)
                    if relative_error <= 0.01:  # 1%误差
                        return True, "Correct!"
                    else:
                        return False, f"Incorrect. Expected {correct_value}, got {user_value}"
            
            # 检查百分比
            if "%" in correct_answer or "percent" in correct_answer.lower():
                success, feedback = self._check_percentage_answer(action, correct_answer)
                if success:
                    return success, feedback
            
            # 文本答案的模糊匹配
            if self._fuzzy_text_match(action, correct_answer):
                return True, "Correct!"
        
        return success, feedback
    
    def _is_fraction_answer(self, answer: str) -> bool:
        """判断是否为分数答案"""
        return "/" in answer and re.search(r'\d+/\d+', answer)
    
    def _check_fraction_answer(self, user_answer: str, correct_answer: str) -> Tuple[bool, str]:
        """检查分数答案"""
        try:
            # 提取用户答案中的分数
            user_fraction = self._extract_fraction(user_answer)
            correct_fraction = self._extract_fraction(correct_answer)
            
            if user_fraction and correct_fraction:
                # 比较分数值
                if abs(float(user_fraction) - float(correct_fraction)) < 0.001:
                    return True, "Correct fraction!"
                
                # 检查是否是等价分数
                if user_fraction == correct_fraction:
                    return True, "Correct fraction!"
            
            return False, f"Incorrect fraction. Expected {correct_answer}, got {user_answer}"
            
        except Exception as e:
            logger.debug(f"Fraction parsing error: {e}")
            return False, f"Invalid fraction format. Expected {correct_answer}"
    
    def _extract_fraction(self, text: str) -> Optional[Fraction]:
        """从文本中提取分数"""
        fraction_match = re.search(r'(\d+)/(\d+)', text)
        if fraction_match:
            try:
                numerator = int(fraction_match.group(1))
                denominator = int(fraction_match.group(2))
                return Fraction(numerator, denominator)
            except:
                pass
        return None
    
    def _check_percentage_answer(self, user_answer: str, correct_answer: str) -> Tuple[bool, str]:
        """检查百分比答案"""
        user_percent = self._extract_percentage(user_answer)
        correct_percent = self._extract_percentage(correct_answer)
        
        if user_percent is not None and correct_percent is not None:
            if abs(user_percent - correct_percent) <= 0.1:  # 0.1%误差
                return True, "Correct percentage!"
            else:
                return False, f"Incorrect. Expected {correct_percent}%, got {user_percent}%"
        
        return False, "Invalid percentage format"
    
    def _extract_percentage(self, text: str) -> Optional[float]:
        """从文本中提取百分比"""
        # 移除%符号进行数字提取
        clean_text = text.replace('%', '').replace('percent', '')
        number = self._extract_number(clean_text)
        return number
    
    def _extract_number(self, text: str) -> Optional[float]:
        """从文本中提取数字"""
        if not text:
            return None
        
        # 清理文本
        text = text.lower().strip()
        
        # 移除常见的非数字字符
        text = re.sub(r'[^\d.,\-/]', ' ', text)
        
        # 处理分数
        fraction_match = re.search(r'(\d+)/(\d+)', text)
        if fraction_match:
            try:
                numerator = float(fraction_match.group(1))
                denominator = float(fraction_match.group(2))
                if denominator != 0:
                    return numerator / denominator
            except:
                pass
        
        # 处理小数
        decimal_match = re.search(r'-?\d+\.?\d*', text)
        if decimal_match:
            try:
                return float(decimal_match.group())
            except:
                pass
        
        return None
    
    def _fuzzy_text_match(self, user_answer: str, correct_answer: str) -> bool:
        """模糊文本匹配"""
        user_clean = re.sub(r'[^\w]', '', user_answer.lower())
        correct_clean = re.sub(r'[^\w]', '', correct_answer.lower())
        
        # 精确匹配
        if user_clean == correct_clean:
            return True
        
        # 包含匹配
        if user_clean in correct_clean or correct_clean in user_clean:
            return True
        
        return False
    
    def validate(
        self,
        chat_history: List[Dict],
        observation: Any,
        full_history: Optional[List[Any]] = None
    ) -> Tuple[float, bool, str, Dict[str, Any]]:
        """
        验证TabMWP任务执行情况
        """
        # 调用父类验证
        reward, done, message, info = super().validate(
            chat_history, observation, full_history
        )
        
        # 添加TabMWP特定的信息
        info["problem_type"] = self._detect_problem_type()
        info["calculation_type"] = self._detect_calculation_type()
        info["table_type"] = self._detect_table_type()
        info["requires_multi_step"] = self._requires_multi_step()
        info["statistical_concept"] = self._detect_statistical_concept()
        
        # 如果是数值答案，添加提取的数值信息
        if info.get("answer_provided"):
            extracted_value = self._extract_number(str(info["answer_provided"]))
            if extracted_value is not None:
                info["extracted_numerical_value"] = extracted_value
        
        return reward, done, message, info
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取TabMWP特定的指标"""
        metrics = super().get_metrics()
        
        metrics.update({
            "problem_type": self._detect_problem_type(),
            "calculation_type": self._detect_calculation_type(),
            "table_type": self._detect_table_type(),
            "requires_multi_step": self._requires_multi_step(),
            "statistical_concept": self._detect_statistical_concept(),
            "grade_level": self.metadata.get("grade_level", 0) if self.metadata else 0,
            "answer_format": self.metadata.get("answer_format", "unknown") if self.metadata else "unknown",
            "has_table_data": bool(self.metadata.get("table_data")) if self.metadata else False,
            "problem_complexity": self._assess_complexity()
        })
        
        return metrics
    
    def _assess_complexity(self) -> str:
        """评估问题复杂度"""
        if not self.question:
            return "unknown"
        
        complexity_score = 0
        
        # 基于年级水平
        grade = self.metadata.get("grade_level", 0) if self.metadata else 0
        if grade >= 7:
            complexity_score += 2
        elif grade >= 5:
            complexity_score += 1
        
        # 基于问题类型
        problem_type = self._detect_problem_type()
        if problem_type in ["statistics", "fraction", "function_analysis"]:
            complexity_score += 2
        elif problem_type in ["calculation", "comparison"]:
            complexity_score += 1
        
        # 基于多步骤要求
        if self._requires_multi_step():
            complexity_score += 2
        
        # 基于统计概念
        stat_concept = self._detect_statistical_concept()
        if stat_concept != "none":
            complexity_score += 1
        
        # 分类
        if complexity_score >= 4:
            return "high"
        elif complexity_score >= 2:
            return "medium"
        else:
            return "low"
    
    def get_hint(self) -> str:
        """提供任务特定的提示"""
        hints = []
        
        # 基于问题类型的提示
        problem_type = self._detect_problem_type()
        if problem_type == "counting":
            hints.append("Count carefully and make sure you include all items that meet the criteria")
        elif problem_type == "comparison":
            hints.append("Compare all values systematically to find the highest/lowest")
        elif problem_type == "fraction":
            hints.append("Remember to simplify your fraction to its lowest terms")
        elif problem_type == "statistics":
            hints.append("Use the appropriate statistical formula and show your calculations")
        
        # 基于表格类型的提示
        table_type = self._detect_table_type()
        if table_type == "stem_and_leaf":
            hints.append("In stem-and-leaf plots, combine stem and leaf to get the full number")
        elif table_type == "frequency":
            hints.append("Add up all frequencies to get totals")
        
        # 通用提示
        if not hints:
            hints.append("Read the table carefully and extract the relevant data")
        
        return ". ".join(hints)