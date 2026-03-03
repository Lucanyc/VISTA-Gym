#!/usr/bin/env python3
"""
MapQA Task implementation for VLM Gym
Handles map-based visual question answering with various question types
"""

from typing import Tuple, Dict, Any, List, Optional
import re
import ast

from .vision_qa_task import VisionQATask


class MapQATask(VisionQATask):
    """
    MapQA 特定任务
    
    专门处理地图相关的视觉问答任务，支持多种问题类型：
    - relational: 关系比较问题（如"Does X have higher value than Y?"）
    - retrieval: 信息检索问题（如"Which states have value in range X-Y?"）
    
    支持多种答案类型：
    - Yes/No: 二元判断
    - List: 州/地区列表
    - Range: 数值范围
    - Single entity: 单个州/地区
    """
    
    # MapQA的问题类型
    QUESTION_TYPES = {
        'relational': 'Comparative questions about map values',
        'retrieval': 'Information retrieval from maps',
        'spatial': 'Spatial relationship questions',
        'counting': 'Counting elements on maps'
    }
    
    # 问题模板分类
    TEMPLATE_CATEGORIES = {
        'relational': [
            'relational_0', 'relational_1', 'relational_2', 'relational_3',
            'relational_4', 'relational_5', 'relational_6', 'relational_7',
            'relational_8', 'relational_9', 'relational_10', 'relational_11',
            'relational_12', 'relational_13', 'relational_14', 'relational_15',
            'relational_16', 'relational_17', 'relational_18', 'relational_19',
            'relational_20'
        ],
        'retrieval': [
            'retrieval_0', 'retrieval_1', 'retrieval_2', 'retrieval_3'
        ]
    }
    
    # 美国地区分类
    US_REGIONS = {
        'Northeast': ['Maine', 'New Hampshire', 'Vermont', 'Massachusetts', 'Rhode Island', 
                      'Connecticut', 'New York', 'New Jersey', 'Pennsylvania'],
        'South': ['Delaware', 'Maryland', 'Virginia', 'West Virginia', 'Kentucky', 
                  'North Carolina', 'South Carolina', 'Tennessee', 'Georgia', 'Florida',
                  'Alabama', 'Mississippi', 'Louisiana', 'Arkansas', 'Texas', 'Oklahoma'],
        'Midwest': ['Ohio', 'Indiana', 'Illinois', 'Michigan', 'Wisconsin', 'Minnesota',
                    'Iowa', 'Missouri', 'North Dakota', 'South Dakota', 'Nebraska', 'Kansas'],
        'West': ['Montana', 'Idaho', 'Wyoming', 'Colorado', 'New Mexico', 'Arizona', 
                 'Utah', 'Nevada', 'Washington', 'Oregon', 'California', 'Alaska', 'Hawaii']
    }
    
    @classmethod
    def get_task_id(cls) -> str:
        """获取任务类型ID"""
        return "vlm-gym.mapqa"
    
    def setup(self) -> Tuple[str, Dict[str, Any]]:
        """设置MapQA特定的任务"""
        # 调用父类setup
        task_goal, task_info = super().setup()
        
        # 从metadata中提取MapQA特定信息
        self.question_type = self.metadata.get('question_type', 'unknown')
        self.question_template_id = self.metadata.get('question_template_id', '')
        self.image_name = self.metadata.get('image_name', '')
        self.question_id = self.metadata.get('question_id', -1)
        
        # Oracle信息（如果有）
        self.oracle_question = self.metadata.get('oracle_delexicalized_question', '')
        self.oracle_answer = self.metadata.get('oracle_delexicalized_answer', [])
        
        # 分析答案类型
        self.answer_type = self._classify_answer_type()
        
        # 判断任务特征
        self.is_relational = self.question_type == 'relational'
        self.is_retrieval = self.question_type == 'retrieval'
        self.involves_region = self._check_involves_region()
        self.involves_comparison = self._check_involves_comparison()
        self.involves_range = self._check_involves_range()
        
        # 添加MapQA特定的处理
        task_info["question_type"] = self.question_type
        task_info["question_template_id"] = self.question_template_id
        task_info["answer_type"] = self.answer_type
        task_info["is_relational"] = self.is_relational
        task_info["is_retrieval"] = self.is_retrieval
        task_info["involves_region"] = self.involves_region
        task_info["involves_comparison"] = self.involves_comparison
        task_info["involves_range"] = self.involves_range
        task_info["image_name"] = self.image_name
        task_info["dataset"] = "mapqa"
        
        # 修改任务目标以包含MapQA特定指导
        enhanced_goal = self._enhance_task_goal(task_goal)
        
        return enhanced_goal, task_info
    
    def _enhance_task_goal(self, base_goal: str) -> str:
        """增强任务目标描述，添加MapQA特定的指导"""
        enhanced_parts = [base_goal]
        
        # 根据问题类型添加特定指导
        if self.is_relational:
            enhanced_parts.append("\n**This is a relational/comparison question about map data.**")
            enhanced_parts.append("Steps to solve:")
            enhanced_parts.append("1. Identify the entities (states/regions) being compared")
            enhanced_parts.append("2. Locate their values on the map (check the legend)")
            enhanced_parts.append("3. Perform the comparison as asked")
            
            if self.involves_region:
                enhanced_parts.append("4. Pay attention to the regional boundaries mentioned")
                
        elif self.is_retrieval:
            enhanced_parts.append("\n**This is a retrieval question requiring you to find specific information from the map.**")
            enhanced_parts.append("Steps to solve:")
            enhanced_parts.append("1. Understand what information is being requested")
            enhanced_parts.append("2. Check the map legend for value ranges/categories")
            enhanced_parts.append("3. Identify all entities that match the criteria")
            enhanced_parts.append("4. List them completely and accurately")
        
        # 根据答案类型添加格式指导
        if self.answer_type == 'yes_no':
            enhanced_parts.append("\n**Answer Format**: Please answer with 'Yes' or 'No'.")
        elif self.answer_type == 'list':
            enhanced_parts.append("\n**Answer Format**: Provide a complete list of states/entities.")
            enhanced_parts.append("Example format: ['State1', 'State2', 'State3']")
        elif self.answer_type == 'range':
            enhanced_parts.append("\n**Answer Format**: Provide the value range from the map legend.")
            enhanced_parts.append("Example format: 'X-Y' or 'X to Y'")
        elif self.answer_type == 'single_entity':
            enhanced_parts.append("\n**Answer Format**: Name the specific state or entity.")
        
        # 添加地图分析通用提示
        enhanced_parts.append("\n**Map Analysis Tips:**")
        enhanced_parts.append("- First examine the map title and what it represents")
        enhanced_parts.append("- Check the legend carefully for value ranges and color coding")
        enhanced_parts.append("- Pay attention to geographical boundaries and regions")
        enhanced_parts.append("- Be precise when reading values from the color/pattern coding")
        
        return "\n".join(enhanced_parts)
    
    def _classify_answer_type(self) -> str:
        """分类答案类型"""
        if not self.answer:
            return 'unknown'
        
        answer_str = str(self.answer).strip()
        
        # Yes/No答案
        if answer_str.lower() in ['yes', 'no']:
            return 'yes_no'
        
        # 列表答案 (包含多个州)
        if answer_str.startswith('[') and answer_str.endswith(']'):
            try:
                parsed = ast.literal_eval(answer_str)
                if isinstance(parsed, list) and len(parsed) > 1:
                    return 'list'
                elif isinstance(parsed, list) and len(parsed) == 1:
                    return 'single_entity'
            except:
                pass
        
        # 范围答案 (包含数字和连字符)
        if '-' in answer_str and any(char.isdigit() for char in answer_str):
            return 'range'
        
        # 其他情况视为单个实体
        return 'single_entity'
    
    def _check_involves_region(self) -> bool:
        """检查问题是否涉及地区"""
        if not self.question:
            return False
        
        question_lower = self.question.lower()
        regions = ['northeast', 'south', 'midwest', 'west', 'usa', 'united states']
        return any(region in question_lower for region in regions)
    
    def _check_involves_comparison(self) -> bool:
        """检查问题是否涉及比较"""
        if not self.question:
            return False
        
        question_lower = self.question.lower()
        comparison_words = ['higher', 'lower', 'highest', 'lowest', 'greater', 'less', 
                           'more', 'fewer', 'maximum', 'minimum', 'compare']
        return any(word in question_lower for word in comparison_words)
    
    def _check_involves_range(self) -> bool:
        """检查问题是否涉及范围"""
        if not self.question:
            return False
        
        question_lower = self.question.lower()
        return 'range' in question_lower or 'between' in question_lower
    
    def check_success(self, action: Any) -> Tuple[bool, str]:
        """
        检查MapQA答案
        
        对于不同类型的答案采用不同的匹配策略
        """
        if not action:
            return False, "No answer provided"
        
        # 清理答案格式
        action_str = str(action).strip()
        correct_answer = str(self.answer).strip()
        
        # 1. Yes/No答案
        if self.answer_type == 'yes_no':
            user_answer = self._extract_yes_no(action_str)
            if user_answer and user_answer.lower() == correct_answer.lower():
                return True, f"Correct! The answer is {correct_answer}"
            elif user_answer:
                return False, f"Incorrect. The correct answer is {correct_answer}"
            else:
                return False, "Please provide a clear Yes or No answer"
        
        # 2. 列表答案
        elif self.answer_type == 'list':
            success, message = self._check_list_answer(action_str, correct_answer)
            return success, message
        
        # 3. 范围答案
        elif self.answer_type == 'range':
            success, message = self._check_range_answer(action_str, correct_answer)
            return success, message
        
        # 4. 单个实体答案
        elif self.answer_type == 'single_entity':
            success, message = self._check_entity_answer(action_str, correct_answer)
            return success, message
        
        # 默认：文本匹配
        if action_str.lower() == correct_answer.lower():
            return True, f"Correct! The answer is {correct_answer}"
        
        return False, f"Incorrect. The correct answer is {correct_answer}"
    
    def _extract_yes_no(self, text: str) -> Optional[str]:
        """从文本中提取Yes/No答案"""
        text_lower = text.lower().strip()
        
        # 直接匹配
        if text_lower in ['yes', 'no']:
            return text_lower
        
        # 匹配包含yes/no的短语
        if 'yes' in text_lower and 'no' not in text_lower:
            return 'yes'
        elif 'no' in text_lower and 'yes' not in text_lower:
            return 'no'
        
        return None
    
    def _check_list_answer(self, user_answer: str, correct_answer: str) -> Tuple[bool, str]:
        """检查列表类型答案"""
        try:
            # 解析正确答案
            correct_list = ast.literal_eval(correct_answer)
            if not isinstance(correct_list, list):
                correct_list = [correct_list]
            correct_set = set(s.lower() for s in correct_list)
            
            # 尝试解析用户答案
            user_list = None
            
            # 尝试直接解析为列表
            try:
                user_list = ast.literal_eval(user_answer)
                if not isinstance(user_list, list):
                    user_list = [user_list]
            except:
                # 尝试从文本中提取州名
                user_list = self._extract_states_from_text(user_answer)
            
            if user_list:
                user_set = set(s.lower() for s in user_list)
                
                # 检查是否完全匹配
                if user_set == correct_set:
                    return True, f"Correct! You identified all {len(correct_set)} states."
                
                # 计算准确率
                missing = correct_set - user_set
                extra = user_set - correct_set
                correct_count = len(correct_set & user_set)
                
                if correct_count > 0:
                    accuracy = correct_count / len(correct_set)
                    if accuracy >= 0.9:  # 90%以上算正确
                        return True, f"Mostly correct! You got {correct_count}/{len(correct_set)} states."
                    else:
                        feedback = f"Partial credit: {correct_count}/{len(correct_set)} states correct."
                        if missing:
                            feedback += f" Missing: {', '.join(sorted(missing)[:3])}"
                            if len(missing) > 3:
                                feedback += f" and {len(missing)-3} more"
                        return False, feedback
                else:
                    return False, f"Incorrect. Expected {len(correct_set)} states."
            
        except Exception as e:
            pass
        
        return False, f"Could not parse your answer. Expected format: {correct_answer}"
    
    def _check_range_answer(self, user_answer: str, correct_answer: str) -> Tuple[bool, str]:
        """检查范围类型答案"""
        # 标准化范围格式
        correct_normalized = correct_answer.replace(' ', '').replace('to', '-')
        user_normalized = user_answer.replace(' ', '').replace('to', '-')
        
        # 提取数字
        correct_numbers = re.findall(r'[\d,]+', correct_normalized)
        user_numbers = re.findall(r'[\d,]+', user_normalized)
        
        if len(correct_numbers) == 2 and len(user_numbers) == 2:
            # 比较数字（移除逗号）
            correct_nums = [int(n.replace(',', '')) for n in correct_numbers]
            user_nums = [int(n.replace(',', '')) for n in user_numbers]
            
            # 检查是否匹配（顺序可能不同）
            if set(correct_nums) == set(user_nums):
                return True, f"Correct! The range is {correct_answer}"
        
        # 直接文本匹配
        if user_normalized == correct_normalized:
            return True, f"Correct! The range is {correct_answer}"
        
        return False, f"Incorrect. The correct range is {correct_answer}"
    
    def _check_entity_answer(self, user_answer: str, correct_answer: str) -> Tuple[bool, str]:
        """检查单个实体答案"""
        # 清理格式
        correct_clean = correct_answer.strip("[]'\"").lower()
        user_clean = user_answer.strip("[]'\"").lower()
        
        # 直接匹配
        if user_clean == correct_clean:
            return True, f"Correct! The answer is {correct_answer}"
        
        # 检查是否包含正确答案
        if correct_clean in user_clean:
            return True, f"Correct! You identified {correct_answer}"
        
        # 检查常见的州名变体
        if self._match_state_variants(user_clean, correct_clean):
            return True, f"Correct! The answer is {correct_answer}"
        
        return False, f"Incorrect. The correct answer is {correct_answer}"
    
    def _extract_states_from_text(self, text: str) -> List[str]:
        """从文本中提取州名"""
        states = []
        
        # 获取所有美国州名
        all_states = set()
        for region_states in self.US_REGIONS.values():
            all_states.update(region_states)
        
        # 在文本中查找州名
        text_lower = text.lower()
        for state in all_states:
            if state.lower() in text_lower:
                states.append(state)
        
        return states
    
    def _match_state_variants(self, user_state: str, correct_state: str) -> bool:
        """匹配州名的不同变体"""
        # 处理常见的缩写和变体
        state_variants = {
            'ny': 'new york',
            'nj': 'new jersey',
            'pa': 'pennsylvania',
            'ca': 'california',
            'tx': 'texas',
            'fl': 'florida',
            # 可以添加更多
        }
        
        # 检查缩写
        for abbr, full in state_variants.items():
            if (user_state == abbr and correct_state == full) or \
               (user_state == full and correct_state == abbr):
                return True
        
        return False
    
    def validate(
        self,
        chat_history: List[Dict],
        observation: Any,
        full_history: Optional[List[Any]] = None
    ) -> Tuple[float, bool, str, Dict[str, Any]]:
        """
        验证MapQA任务执行情况
        """
        # 调用父类验证
        reward, done, message, info = super().validate(
            chat_history, observation, full_history
        )
        
        # 添加MapQA特定的信息
        info["question_type"] = self.question_type
        info["question_template_id"] = self.question_template_id
        info["answer_type"] = self.answer_type
        info["is_relational"] = self.is_relational
        info["is_retrieval"] = self.is_retrieval
        info["involves_region"] = self.involves_region
        info["involves_comparison"] = self.involves_comparison
        info["involves_range"] = self.involves_range
        info["image_name"] = self.image_name
        
        # 分析答案质量
        if info.get("answer_provided"):
            provided_answer = str(info["answer_provided"])
            
            # 对于列表答案，计算覆盖率
            if self.answer_type == 'list':
                try:
                    correct_list = ast.literal_eval(str(self.answer))
                    user_list = self._extract_states_from_text(provided_answer)
                    if user_list and correct_list:
                        coverage = len(set(user_list) & set(correct_list)) / len(correct_list)
                        info["answer_coverage"] = coverage
                except:
                    pass
            
            # 检查答案是否过于冗长
            if len(provided_answer) > 500:
                info["verbose_answer"] = True
        
        return reward, done, message, info
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取MapQA特定的指标"""
        metrics = super().get_metrics()
        
        metrics.update({
            "question_type": self.question_type,
            "question_template_id": self.question_template_id,
            "answer_type": self.answer_type,
            "is_relational": self.is_relational,
            "is_retrieval": self.is_retrieval,
            "involves_region": self.involves_region,
            "involves_comparison": self.involves_comparison,
            "involves_range": self.involves_range,
            "image_name": self.image_name,
            "question_category": self._get_question_category(),
            "difficulty_factors": self.get_difficulty_factors()
        })
        
        return metrics
    
    def _get_question_category(self) -> str:
        """获取问题的大类别"""
        for category, templates in self.TEMPLATE_CATEGORIES.items():
            if self.question_template_id in templates:
                return category
        return 'other'
    
    def get_difficulty_factors(self) -> List[str]:
        """获取影响难度的因素"""
        factors = []
        
        # 问题类型相关
        if self.is_retrieval and self.answer_type == 'list':
            factors.append("multiple_entities_retrieval")
        
        if self.involves_comparison:
            factors.append("requires_value_comparison")
        
        if self.involves_region:
            factors.append("regional_knowledge_required")
        
        if self.involves_range:
            factors.append("range_identification")
        
        # 答案类型相关
        if self.answer_type == 'list':
            factors.append("comprehensive_listing")
        elif self.answer_type == 'range':
            factors.append("legend_reading")
        
        # 问题长度
        if self.question and len(self.question) > 100:
            factors.append("complex_question")
        
        return factors
    
    def get_region_info(self) -> Dict[str, List[str]]:
        """获取地区信息（辅助方法）"""
        return self.US_REGIONS
    
    def get_hint(self) -> str:
        """获取问题提示（用于辅助）"""
        hints = []
        
        if self.is_relational:
            hints.append("Compare the values of different states/regions using the map legend.")
        
        if self.is_retrieval:
            hints.append("Identify all entities that match the given criteria.")
        
        if self.involves_region:
            hints.append("Focus on the specific geographic region mentioned.")
        
        if self.answer_type == 'list':
            hints.append("Make sure to list ALL matching states.")
        elif self.answer_type == 'range':
            hints.append("Read the exact range from the map legend.")
        
        return " ".join(hints) if hints else "Analyze the map carefully to answer the question."