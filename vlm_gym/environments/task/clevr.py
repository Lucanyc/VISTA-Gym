#!/usr/bin/env python3
"""
CLEVR Task implementation for VLM Gym
Based on the CLEVR paper specifications
"""

from typing import Tuple, Dict, Any, List, Optional
import re
from collections import Counter

from .vision_qa_task import VisionQATask


class CLEVRTask(VisionQATask):
    """
    CLEVR 特定任务
    
    专门处理组合式视觉推理任务，包括：
    - 对象计数
    - 属性查询（颜色、形状、大小、材质）
    - 存在性判断
    - 属性比较
    - 空间关系推理
    """
    
    # CLEVR属性的有效值（根据论文定义）
    VALID_COLORS = ["red", "blue", "green", "purple", "yellow", "cyan", "gray", "brown"]
    VALID_SHAPES = ["cube", "sphere", "cylinder"]
    VALID_SIZES = ["small", "large"]
    VALID_MATERIALS = ["rubber", "metal"]  # 主要使用material术语
    VALID_RELATIONS = ["left", "right", "behind", "in front"]
    
    # 材质的同义词映射
    MATERIAL_SYNONYMS = {
        "rubber": ["matte"],
        "metal": ["shiny", "metallic"]
    }
    
    # 形状的同义词映射
    SHAPE_SYNONYMS = {
        "sphere": ["ball"],
        "cube": ["block"]
    }
    
    @classmethod
    def get_task_id(cls) -> str:
        """获取任务类型ID"""
        return "vlm-gym.clevr"
    
    def setup(self) -> Tuple[str, Dict[str, Any]]:
        """设置CLEVR特定的任务"""
        # 调用父类setup
        task_goal, task_info = super().setup()
        
        # 添加CLEVR特定的处理
        task_info["question_type"] = self._classify_question_type()
        task_info["answer_type"] = self._classify_answer_type()
        task_info["attributes_involved"] = self._detect_attributes()
        task_info["relationship_type"] = self._detect_relationship_type()
        task_info["question_topology"] = self._detect_question_topology()
        task_info["spatial_reasoning"] = self._requires_spatial_reasoning()
        task_info["comparison_needed"] = self._requires_comparison()
        task_info["dataset"] = "clevr"
        
        # 从metadata中提取program信息（如果有）
        if hasattr(self, 'metadata') and self.metadata:
            task_info["program_length"] = self.metadata.get("program_length", 0)
            task_info["program_depth"] = self.metadata.get("program_depth", 0)
            task_info["program_functions"] = self.metadata.get("program_functions", [])
            task_info["question_family"] = self.metadata.get("question_family", -1)
            task_info["has_program"] = bool(self.metadata.get("program"))
            
            # 分析程序结构
            program = self.metadata.get("program", [])
            if program:
                task_info["program_structure"] = self._analyze_program_structure(program)
        
        # 修改任务目标以包含CLEVR特定指导
        enhanced_goal = task_goal
        
        # 根据问题类型添加特定提示
        question_type = task_info["question_type"]
        if question_type == "count":
            enhanced_goal += "\n\nNote: This is a counting question. Please count all objects that match the criteria."
        elif question_type == "exist":
            enhanced_goal += "\n\nNote: This is an existence question. Answer with 'yes' or 'no'."
        elif question_type.startswith("query_"):
            enhanced_goal += "\n\nNote: This question asks about a specific attribute. Identify the object and its property."
        elif question_type in ["less_than", "greater_than", "equal_integer"]:
            enhanced_goal += "\n\nNote: This requires comparing quantities. Count carefully and compare."
        elif question_type == "compare_attribute":
            enhanced_goal += "\n\nNote: This requires comparing object attributes. Check if they are the same."
        
        if task_info["relationship_type"] == "spatial":
            enhanced_goal += "\nPay attention to spatial relationships (left, right, front, behind)."
        elif task_info["relationship_type"] == "same-attribute":
            enhanced_goal += "\nIdentify objects with the same attribute values."
        
        # 添加CLEVR特定的分析提示
        enhanced_goal += "\n\nWhen analyzing the scene, please:"
        enhanced_goal += "\n- Identify all objects and their properties (shape, color, size, material)"
        enhanced_goal += "\n- Note spatial relationships between objects"
        enhanced_goal += "\n- Follow the logical steps needed to answer the question"
        enhanced_goal += "\n- Be precise with object descriptions"
        
        # 添加属性值提示
        enhanced_goal += "\n\nCLEVR objects have these exact attributes:"
        enhanced_goal += f"\n- Shapes: {', '.join(self.VALID_SHAPES)}"
        enhanced_goal += f"\n- Colors: {', '.join(self.VALID_COLORS)}"
        enhanced_goal += f"\n- Sizes: {', '.join(self.VALID_SIZES)}"
        enhanced_goal += "\n- Materials: rubber (matte), metal (shiny)"
        
        return enhanced_goal, task_info
    
    def _classify_question_type(self) -> str:
        """分类CLEVR问题类型 - 根据论文Figure 3"""
        if not self.question:
            return "unknown"
        
        question_lower = self.question.lower()
        
        # Query Attribute (包含4个子类型)
        if question_lower.startswith("what"):
            if "color" in question_lower:
                return "query_color"
            elif "shape" in question_lower:
                return "query_shape"
            elif "size" in question_lower:
                return "query_size"
            elif "material" in question_lower or "made of" in question_lower:
                return "query_material"
            else:
                return "query_attribute"
        
        # Count
        if question_lower.startswith("how many"):
            return "count"
        
        # Exist
        if question_lower.startswith(("is there", "are there")):
            return "exist"
        
        # Equal Integer - 处理"Are there an equal number of..."这种情况
        if "equal number" in question_lower or "same number" in question_lower:
            return "equal_integer"
        
        # Compare Attribute (Equal)
        if "same" in question_lower and any(attr in question_lower for attr in ["color", "shape", "size", "material"]):
            return "compare_attribute"
        
        # Compare Integer (Less/Greater)
        if any(word in question_lower for word in ["fewer", "more", "less", "greater"]):
            if "than" in question_lower:
                if "fewer" in question_lower or "less" in question_lower:
                    return "less_than"
                else:
                    return "greater_than"
        
        # Additional yes/no questions
        if any(question_lower.startswith(prefix) for prefix in ["is the", "are the", "does", "do"]):
            return "yes_no"
        
        return "unknown"
    
    def _classify_answer_type(self) -> str:
        """分类答案类型"""
        if not self.answer:
            return "unknown"
        
        answer_str = str(self.answer).lower()
        
        # Boolean
        if answer_str in ["yes", "no", "true", "false"]:
            return "boolean"
        
        # Integer (counting)
        if answer_str.isdigit():
            return "integer"
        
        # Color attribute
        if answer_str in self.VALID_COLORS:
            return "color"
        
        # Shape attribute
        if answer_str in self.VALID_SHAPES:
            return "shape"
        
        # Size attribute
        if answer_str in self.VALID_SIZES:
            return "size"
        
        # Material attribute - 检查主要术语和同义词
        if answer_str in self.VALID_MATERIALS:
            return "material"
        # 检查同义词
        for material, synonyms in self.MATERIAL_SYNONYMS.items():
            if answer_str in synonyms:
                return "material"
        
        return "other"
    
    def _detect_attributes(self) -> List[str]:
        """检测问题中涉及的属性"""
        if not self.question:
            return []
        
        question_lower = self.question.lower()
        attributes = []
        
        # Size - 只用CLEVR中的词汇
        if any(word in question_lower for word in self.VALID_SIZES):
            attributes.append("size")
        
        # Color
        if any(color in question_lower for color in self.VALID_COLORS):
            attributes.append("color")
        
        # Shape - 包括同义词
        shape_words = self.VALID_SHAPES + ["ball", "block"]
        if any(shape in question_lower for shape in shape_words):
            attributes.append("shape")
        
        # Material - 包括同义词
        material_words = self.VALID_MATERIALS + ["matte", "shiny", "metallic"]
        if any(material in question_lower for material in material_words):
            attributes.append("material")
        
        return list(set(attributes))  # 去重
    
    def _detect_relationship_type(self) -> str:
        """检测关系类型：spatial或same-attribute"""
        if not self.question:
            return "none"
        
        question_lower = self.question.lower()
        
        # Spatial relationships
        if any(rel in question_lower for rel in self.VALID_RELATIONS):
            return "spatial"
        
        # Same-attribute relationships
        if "same" in question_lower and any(attr in question_lower for attr in ["size", "color", "shape", "material"]):
            return "same-attribute"
        
        return "none"
    
    def _detect_question_topology(self) -> str:
        """检测问题拓扑：chain或tree"""
        # 从metadata中的program推断
        if hasattr(self, 'metadata') and self.metadata:
            program = self.metadata.get("program", [])
            if program:
                # 分析程序结构
                # 如果有多个filter操作连接到同一个节点，通常是tree结构
                input_counts = Counter()
                for step in program:
                    if isinstance(step, dict):
                        for inp in step.get("inputs", []):
                            input_counts[inp] += 1
                
                # 如果有节点被多次引用，说明是tree结构
                if any(count > 1 for count in input_counts.values()):
                    return "tree"
        
        # 从问题文本推断
        question_lower = self.question.lower()
        
        # Tree结构的标志
        # 包含logical AND
        if " and " in question_lower and "that" in question_lower:
            return "tree"  # 如 "objects that are X and Y"
        
        # 包含both...and...的问题
        if "both" in question_lower and "and" in question_lower:
            return "tree"
        
        # 包含either...or...的问题
        if "either" in question_lower and "or" in question_lower:
            return "tree"
        
        # 多个独立分支的标志
        if question_lower.count("how many") > 1:
            return "tree"
        
        # 比较问题通常有两个分支
        if "more" in question_lower and "than" in question_lower:
            return "tree"
        if "fewer" in question_lower and "than" in question_lower:
            return "tree"
        
        return "chain"
    
    def _requires_spatial_reasoning(self) -> bool:
        """判断是否需要空间推理"""
        if not self.question:
            return False
        
        # 只使用CLEVR定义的4种空间关系
        question_lower = self.question.lower()
        return any(relation in question_lower for relation in self.VALID_RELATIONS)
    
    def _requires_comparison(self) -> bool:
        """判断是否需要比较"""
        if not self.question:
            return False
        
        comparison_keywords = [
            "same", "different", "equal", "more", "less", "fewer",
            "greater", "smaller", "larger", "compare", "than"
        ]
        
        question_lower = self.question.lower()
        return any(keyword in question_lower for keyword in comparison_keywords)
    
    def _analyze_program_structure(self, program: List[Dict]) -> Dict[str, Any]:
        """分析程序结构，提供调试信息"""
        structure_info = {
            "function_counts": Counter(),
            "max_depth": 0,
            "has_and_operation": False,
            "has_or_operation": False,
            "filter_chain_length": 0
        }
        
        # 统计函数使用
        for step in program:
            if isinstance(step, dict) and "function" in step:
                func_name = step["function"]
                structure_info["function_counts"][func_name] += 1
                
                # 检查逻辑操作
                if func_name == "and":
                    structure_info["has_and_operation"] = True
                elif func_name == "or":
                    structure_info["has_or_operation"] = True
        
        # 计算最大深度和filter链长度
        # （简化实现，实际可能需要更复杂的图分析）
        filter_count = structure_info["function_counts"].get("filter", 0)
        structure_info["filter_chain_length"] = filter_count
        
        return structure_info
    
    def check_success(self, action: Any) -> Tuple[bool, str]:
        """
        检查CLEVR答案
        
        CLEVR答案比较直接，通常是精确匹配
        """
        if not action:
            return False, "No answer provided"
        
        # 清理答案格式
        action = str(action).strip().lower()
        
        # 首先尝试父类的检查
        success, feedback = super().check_success(action)
        
        # 如果父类检查成功，直接返回
        if success:
            return success, feedback
        
        # CLEVR特定的答案处理
        if self.answer:
            correct_answer = str(self.answer).strip().lower()
            
            # 综合同义词字典
            all_synonyms = {}
            all_synonyms.update(self.MATERIAL_SYNONYMS)
            all_synonyms.update(self.SHAPE_SYNONYMS)
            all_synonyms.update({
                "yes": ["true"],
                "no": ["false"]
            })
            
            # 检查同义词
            for key, values in all_synonyms.items():
                if correct_answer == key and action in values:
                    return True, f"Correct! (accepted synonym: {action} for {key})"
                if action == key and correct_answer in values:
                    return True, f"Correct! (accepted synonym: {action} for {correct_answer})"
            
            # 对于数字答案，确保精确匹配
            if correct_answer.isdigit() and action.isdigit():
                if int(correct_answer) == int(action):
                    return True, "Correct!"
                else:
                    return False, f"Incorrect. Expected {correct_answer}, got {action}"
            
            # 对于yes/no答案，处理各种格式
            if correct_answer in ["yes", "no"]:
                # 接受各种肯定/否定表达
                yes_variants = ["yes", "yeah", "yep", "correct", "true", "affirmative"]
                no_variants = ["no", "nope", "incorrect", "false", "negative"]
                
                if correct_answer == "yes" and any(v in action for v in yes_variants):
                    return True, "Correct!"
                if correct_answer == "no" and any(v in action for v in no_variants):
                    return True, "Correct!"
            
            # 检查答案是否包含正确答案（用于处理较长的回答）
            if len(action) > len(correct_answer) and correct_answer in action:
                # 确保不是部分匹配（如 "red" 不应匹配 "reddish"）
                import re
                if re.search(r'\b' + re.escape(correct_answer) + r'\b', action):
                    return True, f"Correct! (found answer '{correct_answer}' in response)"
        
        return False, f"Incorrect. Expected '{self.answer}', got '{action}'"
    
    def validate(
        self,
        chat_history: List[Dict],
        observation: Any,
        full_history: Optional[List[Any]] = None
    ) -> Tuple[float, bool, str, Dict[str, Any]]:
        """
        验证CLEVR任务执行情况
        """
        # 调用父类验证
        reward, done, message, info = super().validate(
            chat_history, observation, full_history
        )
        
        # 添加CLEVR特定的信息
        info["question_type"] = self._classify_question_type()
        info["answer_type"] = self._classify_answer_type()
        info["attributes_involved"] = self._detect_attributes()
        info["relationship_type"] = self._detect_relationship_type()
        info["question_topology"] = self._detect_question_topology()
        info["spatial_reasoning_required"] = self._requires_spatial_reasoning()
        info["comparison_required"] = self._requires_comparison()
        
        # 添加程序复杂度信息（如果有）
        if hasattr(self, 'metadata') and self.metadata:
            info["program_length"] = self.metadata.get("program_length", 0)
            info["program_depth"] = self.metadata.get("program_depth", 0)
            
            # 添加程序结构分析
            program = self.metadata.get("program", [])
            if program:
                info["program_structure"] = self._analyze_program_structure(program)
        
        # 分析答案质量
        if info.get("answer_provided"):
            provided_answer = str(info["answer_provided"]).lower()
            
            # 检查答案格式是否符合预期
            if info["answer_type"] == "boolean" and provided_answer not in ["yes", "no"]:
                info["answer_format_issue"] = "Expected yes/no answer"
            elif info["answer_type"] == "integer" and not provided_answer.isdigit():
                info["answer_format_issue"] = "Expected numerical answer"
            elif info["answer_type"] in ["color", "shape", "size", "material"]:
                valid_values = self._get_valid_attribute_values(info["answer_type"])
                # 扩展检查，包括部分匹配
                found_valid = False
                for valid in valid_values:
                    if valid in provided_answer:
                        found_valid = True
                        break
                if not found_valid:
                    info["answer_format_issue"] = f"Expected one of: {', '.join(valid_values)}"
        
        return reward, done, message, info
    
    def _get_valid_attribute_values(self, attribute_type: str) -> List[str]:
        """获取属性的有效值，包括同义词"""
        valid_values = {
            "color": self.VALID_COLORS,
            "shape": self.VALID_SHAPES + ["ball", "block"],  # 包含常见同义词
            "size": self.VALID_SIZES,
            "material": self.VALID_MATERIALS + ["matte", "shiny", "metallic"]  # 包含所有同义词
        }
        return valid_values.get(attribute_type, [])
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取CLEVR特定的指标"""
        metrics = super().get_metrics()
        
        # 基本指标
        metrics.update({
            "question_type": self._classify_question_type(),
            "answer_type": self._classify_answer_type(),
            "attributes_count": len(self._detect_attributes()),
            "relationship_type": self._detect_relationship_type(),
            "question_topology": self._detect_question_topology(),
            "requires_spatial_reasoning": self._requires_spatial_reasoning(),
            "requires_comparison": self._requires_comparison(),
            "question_complexity": self._assess_complexity()
        })
        
        # 程序相关指标（如果有）
        if hasattr(self, 'metadata') and self.metadata:
            metrics.update({
                "program_length": self.metadata.get("program_length", 0),
                "program_depth": self.metadata.get("program_depth", 0),
                "question_family": self.metadata.get("question_family", -1),
                "has_program": bool(self.metadata.get("program"))
            })
            
            # 分析程序函数使用
            if self.metadata.get("program_functions"):
                function_counts = Counter(self.metadata["program_functions"])
                metrics["most_used_functions"] = dict(function_counts.most_common(5))
            
            # 添加程序结构分析
            program = self.metadata.get("program", [])
            if program:
                metrics["program_structure"] = self._analyze_program_structure(program)
        
        return metrics
    
    def _assess_complexity(self) -> str:
        """评估问题复杂度"""
        if not self.question:
            return "unknown"
        
        complexity_score = 0
        
        # 基于问题类型
        question_type = self._classify_question_type()
        
        # 简单问题类型
        if question_type in ["query_color", "query_shape", "query_size", "query_material", "exist"]:
            complexity_score += 1
        # 中等复杂度
        elif question_type in ["count", "compare_attribute"]:
            complexity_score += 2
        # 高复杂度
        elif question_type in ["less_than", "greater_than", "equal_integer", "unknown"]:
            complexity_score += 3
        
        # 基于属性数量
        attributes_count = len(self._detect_attributes())
        complexity_score += attributes_count
        
        # 基于关系类型
        relationship_type = self._detect_relationship_type()
        if relationship_type == "spatial":
            complexity_score += 2
        elif relationship_type == "same-attribute":
            complexity_score += 2  # same-attribute通常需要更多记忆
        
        # 基于问题拓扑
        if self._detect_question_topology() == "tree":
            complexity_score += 2
        
        # 基于程序长度（如果有）
        if hasattr(self, 'metadata') and self.metadata:
            program_length = self.metadata.get("program_length", 0)
            if program_length > 15:
                complexity_score += 3
            elif program_length > 10:
                complexity_score += 2
            elif program_length > 5:
                complexity_score += 1
        
        # 分类
        if complexity_score >= 8:
            return "high"
        elif complexity_score >= 4:
            return "medium"
        else:
            return "low"
    
    def get_scene_description_prompt(self) -> str:
        """获取场景描述的提示（用于需要先描述场景的模型）"""
        prompt = "Please describe the 3D scene in detail, including:\n"
        prompt += f"1. All objects present (shapes: {', '.join(self.VALID_SHAPES)})\n"
        prompt += "2. Their properties:\n"
        prompt += f"   - Color ({', '.join(self.VALID_COLORS)})\n"
        prompt += f"   - Size ({', '.join(self.VALID_SIZES)})\n"
        prompt += "   - Material (rubber/matte or metal/shiny)\n"
        prompt += f"3. Spatial relationships between objects ({', '.join(self.VALID_RELATIONS)})\n"
        prompt += "4. The overall arrangement of the scene\n"
        prompt += "\nBe precise and systematic in your description."
        return prompt