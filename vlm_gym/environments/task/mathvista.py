# vlm_gym/environments/task/mathvista.py

from typing import Tuple, Dict, Any, List, Optional
import re
import math

from .vision_qa_task import VisionQATask


class MathVistaTask(VisionQATask):
    """
    MathVista 特定任务
    
    专门处理数学和科学视觉推理任务，包括：
    - 物理问题（力学、运动学等）
    - 化学问题（反应、分子结构等）
    - 几何问题（角度、长度、面积等）
    - 测量问题（单位、体积等）
    - 代数问题（方程、变量等）
    - 统计问题（图表、数据分析等）
    """
    
    @classmethod
    def get_task_id(cls) -> str:
        """获取任务类型ID"""
        return "vlm-gym.mathvista"
    
    def setup(self) -> Tuple[str, Dict[str, Any]]:
        """设置MathVista特定的任务"""
        # 调用父类setup
        task_goal, task_info = super().setup()
        
        # 添加MathVista特定的处理
        task_info["subject_type"] = self._detect_subject_type()
        task_info["problem_type"] = self._detect_problem_type()
        task_info["requires_calculation"] = self._requires_calculation()
        task_info["requires_visual_analysis"] = self._requires_visual_analysis()
        task_info["dataset"] = "mathvista"
        
        # 修改任务目标以包含学科特定指导
        enhanced_goal = task_goal
        
        subject_type = task_info["subject_type"]
        
        # 根据学科类型添加特定提示
        if subject_type == "physics":
            enhanced_goal += "\n\nNote: This is a physics problem. Consider relevant physical laws and formulas."
            enhanced_goal += "\n- Identify the physical quantities involved"
            enhanced_goal += "\n- Apply appropriate physics principles"
            enhanced_goal += "\n- Show your calculations with units"
            
        elif subject_type == "chemistry":
            enhanced_goal += "\n\nNote: This is a chemistry problem. Consider chemical properties and reactions."
            enhanced_goal += "\n- Identify chemical elements or compounds"
            enhanced_goal += "\n- Apply chemical principles"
            enhanced_goal += "\n- Consider molecular structures if relevant"
            
        elif subject_type == "geometry":
            enhanced_goal += "\n\nNote: This is a geometry problem. Apply geometric theorems and formulas."
            enhanced_goal += "\n- Identify shapes and their properties"
            enhanced_goal += "\n- Use appropriate geometric formulas"
            enhanced_goal += "\n- Show your geometric reasoning"
            
        elif subject_type == "statistics":
            enhanced_goal += "\n\nNote: This involves data analysis. Carefully read the chart or graph."
            enhanced_goal += "\n- Identify the type of visualization"
            enhanced_goal += "\n- Extract relevant data points"
            enhanced_goal += "\n- Perform necessary calculations"
            
        elif subject_type == "measurement":
            enhanced_goal += "\n\nNote: This is a measurement problem. Pay attention to units and scales."
            enhanced_goal += "\n- Read measurements carefully"
            enhanced_goal += "\n- Convert units if necessary"
            enhanced_goal += "\n- Check your answer's reasonableness"
        
        # 通用数学提示
        enhanced_goal += "\n\nPlease show your step-by-step reasoning and calculations."
        
        return enhanced_goal, task_info
    
    
    def _detect_subject_type(self) -> str:
        """检测学科类型"""
        # 首先检查metadata中的skills
        if hasattr(self, 'metadata') and self.metadata:
            skills = self.metadata.get('skills', [])
            
            if skills:
                # 如果只有一个技能，直接返回（转换为下划线格式）
                if len(skills) == 1:
                    return skills[0].replace(' ', '_')
                
                # 如果有多个技能，选择主要的技能类型
                skill_priority = {
                    'geometry reasoning': 1,
                    'algebraic reasoning': 2,
                    'statistical reasoning': 3,
                    'arithmetic reasoning': 4,
                    'scientific reasoning': 5,
                    'numeric commonsense': 6,
                    'logical reasoning': 7
                }
                
                # 找出优先级最高的技能
                primary_skill = min(skills, key=lambda s: skill_priority.get(s, 999))
                return primary_skill.replace(' ', '_')
            
            # 检查是否有预定义的subject_type
            if 'subject_type' in self.metadata:
                return self.metadata['subject_type'].replace(' ', '_')
        
        # 基于问题内容检测 - 映射到正确的skill分类（下划线格式）
        if not self.question:
            return "arithmetic_reasoning"  # 默认返回算术推理
        
        question_lower = self.question.lower()
        
        # 科学推理指标（包括物理、化学）
        scientific_keywords = [
            'force', 'speed', 'velocity', 'mass', 'spring', 'friction', 
            'energy', 'momentum', 'acceleration', 'motion', 'gravity',
            'pressure', 'temperature', 'heat', 'wave', 'electric', 'magnetic',
            'chemical', 'reaction', 'element', 'compound', 'molecule', 
            'atom', 'ion', 'bond', 'acid', 'base', 'solution', 'concentration'
        ]
        if any(keyword in question_lower for keyword in scientific_keywords):
            return 'scientific_reasoning'
        
        # 几何推理指标
        geometry_keywords = [
            'angle', 'triangle', 'circle', 'square', 'perimeter', 'area',
            'volume', 'radius', 'diameter', 'parallel', 'perpendicular',
            '°', '∠', 'polygon', 'rectangle', 'shape', 'side', 'vertex',
            'diagonal', 'bisector', 'congruent', 'similar'
        ]
        if any(keyword in question_lower for keyword in geometry_keywords):
            return 'geometry_reasoning'
        
        # 代数推理指标
        algebra_keywords = [
            'equation', 'solve for', 'variable', 'x =', 'y =', 'function',
            'expression', 'simplify', 'factor', 'polynomial', 'algebra',
            'quadratic', 'linear', 'exponential', 'logarithm'
        ]
        if any(keyword in question_lower for keyword in algebra_keywords):
            return 'algebraic_reasoning'
        
        # 统计推理指标
        statistics_keywords = [
            'graph', 'chart', 'data', 'average', 'median', 'probability',
            'mean', 'mode', 'distribution', 'frequency', 'percentage',
            'bar chart', 'pie chart', 'histogram', 'scatter plot', 'trend'
        ]
        if any(keyword in question_lower for keyword in statistics_keywords):
            return 'statistical_reasoning'
        
        # 数值常识指标（测量、单位转换等）
        numeric_keywords = [
            'measuring', 'liter', 'meter', 'gram', 'kilogram', 'millimeter',
            'centimeter', 'volume', 'capacity', 'weight', 'length', 'scale',
            'unit', 'convert', 'measurement', 'size', 'distance', 'height'
        ]
        if any(keyword in question_lower for keyword in numeric_keywords):
            return 'numeric_commonsense'
        
        # 逻辑推理指标
        logical_keywords = [
            'true or false', 'is it true', 'correct or incorrect', 'logical',
            'if...then', 'therefore', 'because', 'implies', 'conclusion',
            'reasoning', 'deduce', 'infer'
        ]
        if any(keyword in question_lower for keyword in logical_keywords):
            return 'logical_reasoning'
        
        # 算术推理指标（默认）
        arithmetic_keywords = [
            'calculate', 'compute', 'sum', 'difference', 'product', 'quotient',
            'add', 'subtract', 'multiply', 'divide', 'total', 'how many',
            'how much', '+', '-', '*', '/', '='
        ]
        if any(keyword in question_lower for keyword in arithmetic_keywords):
            return 'arithmetic_reasoning'
        
        # 默认返回算术推理
        return 'arithmetic_reasoning'
        
    
    def _detect_problem_type(self) -> str:
        """检测问题类型"""
        if not self.question:
            return "unknown"
        
        question_lower = self.question.lower()
        
        # 数值计算
        if any(word in question_lower for word in ['calculate', 'compute', 'find the value', 'what is']):
            return 'numerical'
        
        # 比较
        if any(word in question_lower for word in ['compare', 'which is greater', 'which is less', 'difference']):
            return 'comparison'
        
        # 识别
        if any(word in question_lower for word in ['identify', 'what type', 'which', 'name']):
            return 'identification'
        
        # 解释
        if any(word in question_lower for word in ['explain', 'why', 'how', 'describe']):
            return 'explanation'
        
        # 是非判断
        if any(word in question_lower for word in ['true or false', 'is it true', 'correct or incorrect']):
            return 'true_false'
        
        # 多选题（通过choices判断）
        if self.choices:
            return 'multiple_choice'
        
        return 'open_ended'
    
    def _requires_calculation(self) -> bool:
        """判断是否需要计算"""
        if not self.question:
            return False
        
        calculation_indicators = [
            'calculate', 'compute', 'solve', 'find', 'determine',
            'how many', 'how much', 'what is the value',
            '+', '-', '*', '/', '=', 'sum', 'difference', 'product'
        ]
        
        question_lower = self.question.lower()
        return any(indicator in question_lower for indicator in calculation_indicators)
    
    def _requires_visual_analysis(self) -> bool:
        """判断是否需要视觉分析"""
        if not self.question:
            return False
        
        visual_indicators = [
            'graph', 'chart', 'diagram', 'figure', 'image', 'picture',
            'shown', 'depicted', 'illustrated', 'see', 'observe', 'notice'
        ]
        
        question_lower = self.question.lower()
        return any(indicator in question_lower for indicator in visual_indicators)
    
    def check_success(self, action: Any) -> Tuple[bool, str]:
        """
        检查MathVista答案
        
        对于数值答案，允许一定的误差范围
        """
        if not action:
            return False, "No answer provided"
        
        # 清理答案格式
        action = str(action).strip()
        
        # 首先尝试父类的检查
        success, feedback = super().check_success(action)
        
        # 如果父类检查失败，尝试MathVista特定的检查
        if not success and self.answer:
            # 检查数值答案（允许误差）
            user_value = self._extract_number(action)
            correct_value = self._extract_number(str(self.answer))
            
            if user_value is not None and correct_value is not None:
                # 根据问题类型设置不同的容差
                subject_type = self._detect_subject_type()
                
                # 物理问题通常需要更高精度
                if subject_type == 'physics':
                    tolerance = 0.01  # 1%
                # 测量问题可能有更大误差
                elif subject_type == 'measurement':
                    tolerance = 0.05  # 5%
                # 其他数学问题
                else:
                    tolerance = 0.001  # 0.1%
                
                # 计算相对误差
                if correct_value == 0:
                    # 绝对误差检查
                    if abs(user_value - correct_value) <= 0.01:
                        return True, f"Correct! (exact match for zero value)"
                else:
                    relative_error = abs(user_value - correct_value) / abs(correct_value)
                    if relative_error <= tolerance:
                        return True, f"Correct! (within acceptable tolerance: ±{tolerance*100:.1f}%)"
                
                return False, f"Incorrect. Expected {correct_value}, got {user_value}"
            
            # 检查科学记数法
            if self._check_scientific_notation(action, str(self.answer)):
                return True, "Correct! (scientific notation)"
            
            # 检查分数形式
            if self._check_fraction_format(action, str(self.answer)):
                return True, "Correct! (fraction format)"
        
        return success, feedback
    
    def _extract_number(self, text: str) -> Optional[float]:
        """从文本中提取数字（支持科学记数法和分数）"""
        if not text:
            return None
        
        # 清理文本
        text = text.lower().strip()
        
        # 移除常见的干扰字符和单位
        units = ['%', '$', '°', 'degrees', 'meters', 'm', 'cm', 'kg', 'g', 'l', 'ml']
        for unit in units:
            text = text.replace(unit, '')
        
        # 处理百分比
        is_percentage = '%' in text or 'percent' in text
        
        # 尝试提取科学记数法
        sci_pattern = r'-?\d+\.?\d*\s*[×x]\s*10\^?\s*-?\d+'
        sci_match = re.search(sci_pattern, text)
        if sci_match:
            try:
                # 转换科学记数法
                number_str = sci_match.group(0)
                number_str = number_str.replace('×', 'e').replace('x', 'e').replace('10^', 'e').replace(' ', '')
                return float(number_str)
            except ValueError:
                pass
        
        # 尝试提取分数
        fraction_pattern = r'(-?\d+)\s*/\s*(\d+)'
        fraction_match = re.search(fraction_pattern, text)
        if fraction_match:
            try:
                numerator = float(fraction_match.group(1))
                denominator = float(fraction_match.group(2))
                if denominator != 0:
                    return numerator / denominator
            except ValueError:
                pass
        
        # 尝试提取普通数字
        number_pattern = r'-?\d+\.?\d*'
        matches = re.findall(number_pattern, text)
        
        if matches:
            try:
                number = float(matches[0])
                if is_percentage and number > 1:
                    number = number / 100
                return number
            except ValueError:
                pass
        
        return None
    
    def _check_scientific_notation(self, user_answer: str, correct_answer: str) -> bool:
        """检查科学记数法是否等价"""
        user_value = self._extract_number(user_answer)
        correct_value = self._extract_number(correct_answer)
        
        if user_value is None or correct_value is None:
            return False
        
        # 检查是否在误差范围内
        if correct_value != 0:
            relative_error = abs(user_value - correct_value) / abs(correct_value)
            return relative_error <= 0.01  # 1%误差
        else:
            return abs(user_value - correct_value) <= 0.001
    
    def _check_fraction_format(self, user_answer: str, correct_answer: str) -> bool:
        """检查分数格式是否等价"""
        user_value = self._extract_number(user_answer)
        correct_value = self._extract_number(correct_answer)
        
        if user_value is None or correct_value is None:
            return False
        
        # 检查数值是否相等（允许浮点误差）
        return abs(user_value - correct_value) < 1e-9
    
    def validate(
        self,
        chat_history: List[Dict],
        observation: Any,
        full_history: Optional[List[Any]] = None
    ) -> Tuple[float, bool, str, Dict[str, Any]]:
        """
        验证MathVista任务执行情况
        """
        # 调用父类验证
        reward, done, message, info = super().validate(
            chat_history, observation, full_history
        )
        
        # 添加MathVista特定的信息
        info["subject_type"] = self._detect_subject_type()
        info["problem_type"] = self._detect_problem_type()
        info["required_calculation"] = self._requires_calculation()
        info["required_visual_analysis"] = self._requires_visual_analysis()
        
        # 如果是数值答案，添加提取的数值信息
        if info.get("answer_provided"):
            extracted_value = self._extract_number(str(info["answer_provided"]))
            if extracted_value is not None:
                info["extracted_numerical_value"] = extracted_value
        
        return reward, done, message, info
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取MathVista特定的指标"""
        metrics = super().get_metrics()
        
        metrics.update({
            "subject_type": self._detect_subject_type(),
            "problem_type": self._detect_problem_type(),
            "requires_calculation": self._requires_calculation(),
            "requires_visual_analysis": self._requires_visual_analysis(),
            "is_numerical_answer": self._extract_number(str(self.answer)) is not None if self.answer else False,
            "has_scientific_notation": bool(re.search(r'\d+\.?\d*\s*[×x]\s*10\^?\s*-?\d+', str(self.question))),
            "question_complexity": self._assess_complexity()
        })
        
        return metrics
    
    def _assess_complexity(self) -> str:
        """评估问题复杂度"""
        if not self.question:
            return "unknown"
        
        question_lower = self.question.lower()
        complexity_score = 0
        
        # 多步骤指标
        multi_step_indicators = ['then', 'after', 'first', 'second', 'finally', 'next']
        complexity_score += sum(1 for indicator in multi_step_indicators if indicator in question_lower)
        
        # 复杂运算
        if self._requires_calculation():
            complexity_score += 1
        
        # 需要视觉分析
        if self._requires_visual_analysis():
            complexity_score += 1
        
        # 专业术语
        technical_terms = [
            'derivative', 'integral', 'momentum', 'equilibrium', 'reaction',
            'theorem', 'postulate', 'hypothesis', 'coefficient', 'polynomial'
        ]
        complexity_score += sum(0.5 for term in technical_terms if term in question_lower)
        
        # 分类
        if complexity_score >= 3:
            return "high"
        elif complexity_score >= 1.5:
            return "medium"
        else:
            return "low"