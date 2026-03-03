# vlm_gym/environments/task/scienceqa_task.py

from typing import Tuple, Dict, Any, List, Optional
import re

from .vision_qa_task import VisionQATask


class ScienceQATask(VisionQATask):
    """
    ScienceQA 特定任务
    
    专门处理K-12科学教育问题，包括：
    - 自然科学（物理、化学、生物、地球科学）
    - 社会科学（地理、历史、社会研究）
    - 语言艺术
    - 多模态推理（结合图像和文本）
    
    特点：
    - 支持提示（hint）和讲解（lecture）
    - 包含详细的解答过程（solution）
    - 按年级和技能分类
    - 支持科学概念等价表达（如 H2O = water）
    - 严格的答案匹配（无部分分数）
    """
    
    @classmethod
    def get_task_id(cls) -> str:
        """获取任务类型ID"""
        return "vlm-gym.scienceqa"
    
    def setup(self) -> Tuple[str, Dict[str, Any]]:
        """设置ScienceQA特定的任务"""
        # 调用父类setup
        task_goal, task_info = super().setup()
        
        # 添加ScienceQA特定的处理
        task_info["subject"] = self._get_subject()
        task_info["grade_level"] = self._get_grade_level()
        task_info["topic"] = self._get_topic()
        task_info["skills"] = self._get_skills()
        task_info["has_hint"] = bool(getattr(self, 'hint', None))
        task_info["has_lecture"] = bool(getattr(self, 'lecture', None))
        task_info["has_solution"] = bool(getattr(self, 'solution', None))
        task_info["has_diagram"] = self.image_path is not None
        task_info["dataset"] = "scienceqa"
        
        # 构建增强的任务目标
        enhanced_goal = task_goal
        
        # 添加提示信息（如果有）
        if task_info["has_hint"]:
            enhanced_goal += f"\n\nHint: {self.hint}"
        
        # 根据学科类型添加特定指导
        subject = task_info["subject"]
        
        if subject == "natural science":
            enhanced_goal += "\n\nNote: This is a natural science problem. Consider:"
            enhanced_goal += "\n- Scientific principles and laws"
            enhanced_goal += "\n- Observations from the image"
            enhanced_goal += "\n- Cause and effect relationships"
            
            # 根据具体主题添加更详细的指导
            topic = task_info["topic"]
            if "physics" in topic.lower():
                enhanced_goal += "\n- Physical quantities and their relationships"
                enhanced_goal += "\n- Conservation laws if applicable"
            elif "biology" in topic.lower():
                enhanced_goal += "\n- Biological structures and functions"
                enhanced_goal += "\n- Life processes and interactions"
            elif "chemistry" in topic.lower():
                enhanced_goal += "\n- Chemical properties and reactions"
                enhanced_goal += "\n- Molecular structures if shown"
                
        elif subject == "social science":
            enhanced_goal += "\n\nNote: This is a social science problem. Consider:"
            enhanced_goal += "\n- Geographic or historical context"
            enhanced_goal += "\n- Spatial relationships if applicable"
            enhanced_goal += "\n- Cultural or societal factors"
            
            topic = task_info["topic"]
            if "geography" in topic.lower():
                enhanced_goal += "\n- Map reading skills"
                enhanced_goal += "\n- Cardinal directions and scale"
                enhanced_goal += "\n- Geographic features"
        
        # 对于多选题，添加特定指导
        if self.choices:
            enhanced_goal += "\n\nThis is a multiple-choice question. Please:"
            enhanced_goal += "\n- Read all options carefully"
            enhanced_goal += "\n- Select the most accurate answer"
            enhanced_goal += "\n- Provide your answer as the letter (A, B, C, etc.) or the exact text of the correct option"
        
        # 如果有图表，添加图表分析提示
        if task_info["has_diagram"]:
            enhanced_goal += "\n\nAnalyze the image carefully to support your answer."
        
        # 根据技能添加指导
        skills = task_info["skills"]
        if "scientific reasoning" in skills:
            enhanced_goal += "\n\nApply scientific reasoning to solve this problem."
        elif "spatial reasoning" in skills:
            enhanced_goal += "\n\nUse spatial reasoning to analyze relationships."
        
        # 根据年级调整语言
        grade = task_info["grade_level"]
        if grade == "elementary school":
            enhanced_goal += "\n\nThink step by step and explain your reasoning clearly."
        elif grade == "middle school":
            enhanced_goal += "\n\nShow your work and explain your reasoning process."
        elif grade == "high school":
            enhanced_goal += "\n\nProvide detailed analysis and justify your answer."
        
        return enhanced_goal, task_info
    
    def _get_subject(self) -> str:
        """获取学科类型"""
        if hasattr(self, 'metadata') and self.metadata:
            return self.metadata.get('original_subject', 'unknown')
        return "unknown"
    
    def _get_grade_level(self) -> str:
        """获取年级水平"""
        if hasattr(self, 'metadata') and self.metadata:
            return self.metadata.get('grade', 'unknown')
        return "unknown"
    
    def _get_topic(self) -> str:
        """获取具体主题"""
        if hasattr(self, 'metadata') and self.metadata:
            return self.metadata.get('original_topic', 'unknown')
        return "unknown"
    
    def _get_skills(self) -> List[str]:
        """获取所需技能"""
        if hasattr(self, 'metadata') and self.metadata:
            return self.metadata.get('skills', [])
        return []
    
    def check_success(self, action: Any) -> Tuple[bool, str]:
        """
        检查ScienceQA答案
        
        严格匹配答案，但支持科学概念的等价表达
        """
        if not action:
            return False, "No answer provided"
        
        # 清理答案格式
        action = str(action).strip()
        
        # 处理多选题
        if self.choices:
            # 检查是否提供了字母答案 (A, B, C, D...)
            if len(action) == 1 and action.upper().isalpha():
                selected_index = ord(action.upper()) - ord('A')
                if 0 <= selected_index < len(self.choices):
                    selected_choice = self.choices[selected_index]
                    if selected_choice.lower() == str(self.answer).lower():
                        return True, "Correct!"
                    else:
                        return False, f"Incorrect. The correct answer is: {self.answer}"
                else:
                    return False, f"Invalid option. Please select from A-{chr(65 + len(self.choices) - 1)}"
            
            # 检查是否直接提供了选项文本
            action_lower = action.lower()
            answer_lower = str(self.answer).lower()
            
            # 精确匹配（忽略大小写）
            if action_lower == answer_lower:
                return True, "Correct!"
            
            # 检查科学等价表达
            if self._check_scientific_equivalence(action, str(self.answer)):
                return True, "Correct!"
            
            # 对于多选题，不允许部分匹配
            return False, f"Incorrect. The correct answer is: {self.answer}"
        
        # 对于非多选题，使用父类检查
        success, feedback = super().check_success(action)
        
        # 如果父类检查失败，尝试科学等价性检查
        if not success and self.answer:
            if self._check_scientific_equivalence(action, str(self.answer)):
                return True, "Correct!"
        
        return success, feedback
    
    def _check_scientific_equivalence(self, user_answer: str, correct_answer: str) -> bool:
        """检查科学概念的等价性"""
        # 科学同义词映射（小写）
        equivalences = {
            # 化学
            "h2o": ["water"],
            "nacl": ["salt", "sodium chloride", "table salt"],
            "co2": ["carbon dioxide"],
            "o2": ["oxygen", "oxygen gas"],
            "n2": ["nitrogen", "nitrogen gas"],
            "h2": ["hydrogen", "hydrogen gas"],
            "caco3": ["calcium carbonate", "limestone"],
            "hcl": ["hydrochloric acid"],
            "naoh": ["sodium hydroxide", "lye"],
            
            # 物理
            "c": ["speed of light", "light speed"],
            "g": ["gravitational acceleration", "gravity acceleration"],
            "f=ma": ["newton's second law", "force equals mass times acceleration"],
            
            # 生物
            "dna": ["deoxyribonucleic acid", "genetic material"],
            "rna": ["ribonucleic acid"],
            "atp": ["adenosine triphosphate", "energy currency"],
            "photosynthesis": ["making food from sunlight", "converting light to energy"],
            
            # 物理过程
            "evaporation": ["turning into vapor", "becoming gas", "vaporization"],
            "condensation": ["turning into liquid", "becoming water", "liquefaction"],
            "sublimation": ["solid to gas", "direct vaporization"],
            "deposition": ["gas to solid", "direct solidification"],
            
            # 单位
            "m/s": ["meters per second", "metre per second"],
            "km/h": ["kilometers per hour", "kilometres per hour"],
            "°c": ["degrees celsius", "celsius", "centigrade"],
            "°f": ["degrees fahrenheit", "fahrenheit"],
            "k": ["kelvin", "degrees kelvin"],
        }
        
        user_lower = user_answer.lower().strip()
        correct_lower = correct_answer.lower().strip()
        
        # 直接匹配
        if user_lower == correct_lower:
            return True
        
        # 检查等价性（双向）
        for key, values in equivalences.items():
            # 检查 key -> values
            if key == correct_lower:
                if any(val == user_lower for val in values):
                    return True
            # 检查 values -> key
            for val in values:
                if val == correct_lower and key == user_lower:
                    return True
        
        # 检查数值等价（例如 0.5 = 1/2 = 50%）
        if self._check_numerical_equivalence(user_lower, correct_lower):
            return True
        
        return False
    
    def _check_numerical_equivalence(self, user_answer: str, correct_answer: str) -> bool:
        """检查数值等价性"""
        try:
            # 提取数值
            user_num = self._extract_number(user_answer)
            correct_num = self._extract_number(correct_answer)
            
            if user_num is not None and correct_num is not None:
                # 允许极小的浮点误差
                return abs(user_num - correct_num) < 1e-9
        except:
            pass
        
        return False
    
    def _extract_number(self, text: str) -> Optional[float]:
        """从文本中提取数字（支持分数、百分比、科学记数法）"""
        if not text:
            return None
        
        text = text.strip()
        
        # 处理百分比
        if '%' in text:
            match = re.search(r'([\d.]+)\s*%', text)
            if match:
                return float(match.group(1)) / 100
        
        # 处理分数
        if '/' in text:
            match = re.search(r'(\d+)\s*/\s*(\d+)', text)
            if match:
                numerator = float(match.group(1))
                denominator = float(match.group(2))
                if denominator != 0:
                    return numerator / denominator
        
        # 处理科学记数法
        if 'e' in text.lower() or '×10' in text or 'x10' in text:
            # 转换不同格式到标准科学记数法
            sci_text = text.lower()
            sci_text = sci_text.replace('×10^', 'e').replace('x10^', 'e')
            sci_text = sci_text.replace('×10', 'e').replace('x10', 'e')
            match = re.search(r'([\d.]+)e([+-]?\d+)', sci_text)
            if match:
                return float(f"{match.group(1)}e{match.group(2)}")
        
        # 处理普通数字
        match = re.search(r'^-?[\d.]+$', text)
        if match:
            return float(match.group(0))
        
        return None
    
    def validate(
        self,
        chat_history: List[Dict],
        observation: Any,
        full_history: Optional[List[Any]] = None
    ) -> Tuple[float, bool, str, Dict[str, Any]]:
        """
        验证ScienceQA任务执行情况
        
        使用严格的二元评分（正确=1.0，错误=0.0）
        """
        # 调用父类验证
        reward, done, message, info = super().validate(
            chat_history, observation, full_history
        )
        
        # 添加ScienceQA特定的信息
        info["subject"] = self._get_subject()
        info["grade_level"] = self._get_grade_level()
        info["topic"] = self._get_topic()
        info["skills"] = self._get_skills()
        info["hint_used"] = bool(getattr(self, 'hint', None))
        info["lecture_available"] = bool(getattr(self, 'lecture', None))
        info["solution_available"] = bool(getattr(self, 'solution', None))
        
        # 如果有解答，添加到信息中（用于分析）
        if hasattr(self, 'solution') and self.solution:
            info["reference_solution"] = self.solution
        
        # 确保使用二元评分
        if reward > 0 and reward < 1:
            reward = 0.0  # 不允许部分分数
        
        return reward, done, message, info
    
    def get_learning_materials(self) -> Dict[str, Optional[str]]:
        """获取学习材料（讲解和解答）"""
        materials = {}
        
        if hasattr(self, 'lecture') and self.lecture:
            materials['lecture'] = self.lecture
        else:
            materials['lecture'] = None
            
        if hasattr(self, 'solution') and self.solution:
            materials['solution'] = self.solution
        else:
            materials['solution'] = None
            
        return materials
    
    def get_difficulty_estimate(self) -> str:
        """估计问题难度"""
        grade = self._get_grade_level()
        
        # 基于年级的基础难度
        grade_difficulty = {
            'elementary school': 'easy',
            'middle school': 'medium',
            'high school': 'hard',
            'college': 'very hard'
        }
        
        difficulty = grade_difficulty.get(grade, 'medium')
        
        # 根据其他因素调整
        skills = self._get_skills()
        
        # 需要多种技能的问题更难
        if len(skills) > 2:
            if difficulty == 'easy':
                difficulty = 'medium'
            elif difficulty == 'medium':
                difficulty = 'hard'
        
        # 有提示的问题可能原本较难
        if hasattr(self, 'hint') and self.hint:
            if difficulty == 'easy':
                difficulty = 'medium'
        
        return difficulty
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取ScienceQA特定的指标"""
        metrics = super().get_metrics()
        
        metrics.update({
            "subject": self._get_subject(),
            "grade_level": self._get_grade_level(),
            "topic": self._get_topic(),
            "skills": self._get_skills(),
            "difficulty_estimate": self.get_difficulty_estimate(),
            "has_diagram": self.image_path is not None,
            "has_hint": bool(getattr(self, 'hint', None)),
            "has_lecture": bool(getattr(self, 'lecture', None)),
            "has_solution": bool(getattr(self, 'solution', None)),
            "is_multiple_choice": bool(self.choices),
            "num_choices": len(self.choices) if self.choices else 0,
            "question_length": len(self.question) if self.question else 0,
        })
        
        # 添加技能统计
        skills = self._get_skills()
        metrics["num_skills"] = len(skills)
        metrics["primary_skill"] = skills[0] if skills else "none"
        
        # 问题类型细分
        if self.choices:
            metrics["answer_format"] = "multiple_choice"
        else:
            metrics["answer_format"] = "open_ended"
        
        return metrics