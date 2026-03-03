"""
GeoQA Task for VLMGym

Handles Chinese geometry reasoning tasks with multiple choice questions.
"""

from typing import Tuple, Dict, Any, List, Optional
import re

from .vision_qa_task import VisionQATask


class GeoQATask(VisionQATask):
    """
    GeoQA 特定任务
    
    专门处理中文几何推理任务，包括：
    - 几何定理应用
    - 角度计算
    - 平行线性质
    - 三角形性质
    - 圆的性质
    - 图形分析与推理
    """
    
    @classmethod
    def get_task_id(cls) -> str:
        """获取任务类型ID"""
        return "vlm-gym.geoqa"
    
    def setup(self) -> Tuple[str, Dict[str, Any]]:
        """设置GeoQA特定的任务"""
        # 调用父类setup
        task_goal, task_info = super().setup()
        
        # 添加GeoQA特定的处理
        task_info["knowledge_points"] = self._get_knowledge_points()
        task_info["geometry_concepts"] = self._detect_geometry_concepts()
        task_info["requires_theorem"] = self._requires_theorem()
        task_info["problem_type"] = self._classify_problem_type()
        task_info["dataset"] = "geoqa"
        task_info["language"] = "zh"  # Chinese
        
        # 从metadata中提取信息（如果有）
        if hasattr(self, 'metadata') and self.metadata:
            task_info["original_id"] = self.metadata.get("original_id", "")
            task_info["explanation"] = self.metadata.get("explanation", "")
            task_info["formal_points"] = self.metadata.get("knowledge_points", [])
        
        # 修改任务目标以包含GeoQA特定指导
        enhanced_goal = task_goal
        
        # 添加知识点提示
        if task_info["knowledge_points"]:
            points_str = ", ".join(task_info["knowledge_points"])
            enhanced_goal += f"\n\n相关知识点: {points_str}"
        
        # 添加GeoQA特定的提示
        enhanced_goal += "\n\n解题步骤："
        enhanced_goal += "\n1. 仔细观察图形，识别所有几何元素"
        enhanced_goal += "\n2. 找出题目中的已知条件"
        enhanced_goal += "\n3. 确定需要使用的几何定理或性质"
        enhanced_goal += "\n4. 逐步推理得出答案"
        
        # 确保显示选项
        if hasattr(self, 'choices') and self.choices:
            enhanced_goal += f"\n\n请从以下选项中选择: A. {self.choices[0]}, B. {self.choices[1]}, C. {self.choices[2]}, D. {self.choices[3]}"
        
        return enhanced_goal, task_info
    
    def _get_knowledge_points(self) -> List[str]:
        """获取知识点"""
        # 从metadata中获取
        if hasattr(self, 'metadata') and self.metadata:
            return self.metadata.get('knowledge_points', [])
        
        # 从问题中检测
        return self._detect_knowledge_points_from_question()
    
    def _detect_knowledge_points_from_question(self) -> List[str]:
        """从问题文本中检测知识点"""
        if not self.question:
            return []
        
        detected_points = []
        
        # 知识点关键词映射
        knowledge_map = {
            "平行线": ["平行", "∥", "//", "AB∥CD", "DE∥BC"],
            "三角形内角和": ["三角形", "△", "内角和", "180°"],
            "对顶角": ["对顶角", "交点", "相交"],
            "邻补角": ["邻补角", "补角", "互补"],
            "圆周角": ["圆周角", "圆", "弧"],
            "圆心角": ["圆心角", "圆心", "半径"],
            "直角三角形": ["直角三角形", "90°", "Rt△"],
            "等腰三角形": ["等腰", "两边相等", "底角相等"],
            "相似三角形": ["相似", "∽", "比例"],
            "全等三角形": ["全等", "≌"],
            "勾股定理": ["勾股", "直角边", "斜边"],
            "角平分线": ["平分", "角平分线", "平分∠"],
            "垂直": ["垂直", "⊥", "90°"],
            "切线": ["切线", "相切", "切点"],
            "垂径定理": ["垂径", "弦", "垂直平分"]
        }
        
        for point, keywords in knowledge_map.items():
            if any(keyword in self.question for keyword in keywords):
                detected_points.append(point)
        
        return detected_points
    
    def _detect_geometry_concepts(self) -> List[str]:
        """检测涉及的几何概念"""
        if not self.question:
            return []
        
        concepts = []
        
        # 几何图形
        shapes = {
            "三角形": ["三角形", "△"],
            "圆": ["圆", "⊙", "半径", "直径"],
            "四边形": ["四边形", "平行四边形", "矩形", "正方形", "菱形", "梯形"],
            "角": ["∠", "角", "度"],
            "直线": ["直线", "线段", "AB", "CD"],
            "平行线": ["∥", "平行"],
            "垂线": ["⊥", "垂直"]
        }
        
        for concept, keywords in shapes.items():
            if any(keyword in self.question for keyword in keywords):
                concepts.append(concept)
        
        return concepts
    
    def _requires_theorem(self) -> bool:
        """判断是否需要应用几何定理"""
        if not self.question:
            return False
        
        # 需要定理的标志词
        theorem_indicators = [
            "求", "计算", "证明", "判断", "是否",
            "大小是", "等于", "为什么", "理由"
        ]
        
        return any(indicator in self.question for indicator in theorem_indicators)
    
    def _classify_problem_type(self) -> str:
        """分类问题类型"""
        if not self.question:
            return "unknown"
        
        # 角度计算
        if any(word in self.question for word in ["∠", "角", "度", "°"]):
            if "求" in self.question or "大小" in self.question:
                return "angle_calculation"
        
        # 长度计算
        if any(word in self.question for word in ["长度", "长", "距离", "边"]):
            return "length_calculation"
        
        # 面积计算
        if "面积" in self.question:
            return "area_calculation"
        
        # 判断题
        if any(word in self.question for word in ["是否", "判断", "正确"]):
            return "judgment"
        
        # 证明题
        if "证明" in self.question:
            return "proof"
        
        # 位置关系
        if any(word in self.question for word in ["位置", "关系", "平行", "垂直"]):
            return "position_relation"
        
        return "general"
    
    def check_success(self, action: Any) -> Tuple[bool, str]:
        """
        检查GeoQA答案
        
        GeoQA使用字母选项（A、B、C、D）
        """
        if not action:
            return False, "未提供答案"
        
        # 清理答案格式
        action = str(action).strip().upper()
        
        # 提取字母答案（支持各种格式）
        # 例如: "A", "A.", "答案是A", "选A", "(A)"
        letter_match = re.search(r'[A-D]', action)
        if letter_match:
            action = letter_match.group(0)
        
        # 检查答案
        if hasattr(self, 'answer') and self.answer:
            correct_answer = str(self.answer).strip().upper()
            
            if action == correct_answer:
                # 根据知识点给出不同的反馈
                knowledge_points = self._get_knowledge_points()
                if knowledge_points:
                    points_str = "、".join(knowledge_points[:2])  # 最多显示2个知识点
                    return True, f"正确！很好地运用了{points_str}的知识。"
                else:
                    return True, "正确！"
            else:
                # 提供有帮助的错误反馈
                hint = self._generate_hint()
                return False, f"答案不正确。正确答案是 {correct_answer}。{hint}"
        
        return False, "无法验证答案"
    
    def _generate_hint(self) -> str:
        """生成提示信息"""
        knowledge_points = self._get_knowledge_points()
        if knowledge_points:
            return f"提示：这道题需要用到{knowledge_points[0]}的知识。"
        
        problem_type = self._classify_problem_type()
        hints = {
            "angle_calculation": "提示：注意图中的角度关系和已知条件。",
            "length_calculation": "提示：考虑使用相关的几何定理计算长度。",
            "area_calculation": "提示：回顾面积计算公式。",
            "position_relation": "提示：仔细观察图形中的位置关系。",
            "general": "提示：仔细分析题目条件和图形。"
        }
        
        return hints.get(problem_type, "提示：再仔细看看题目和图形。")
    
    def validate(
        self,
        chat_history: List[Dict],
        observation: Any,
        full_history: Optional[List[Any]] = None
    ) -> Tuple[float, bool, str, Dict[str, Any]]:
        """
        验证GeoQA任务执行情况
        """
        # 调用父类验证
        reward, done, message, info = super().validate(
            chat_history, observation, full_history
        )
        
        # 添加GeoQA特定的信息
        info["knowledge_points"] = self._get_knowledge_points()
        info["geometry_concepts"] = self._detect_geometry_concepts()
        info["problem_type"] = self._classify_problem_type()
        info["requires_theorem"] = self._requires_theorem()
        
        # 添加解释信息（如果有）
        if hasattr(self, 'metadata') and self.metadata:
            info["has_explanation"] = bool(self.metadata.get("explanation"))
        
        return reward, done, message, info
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取GeoQA特定的指标"""
        metrics = super().get_metrics()
        
        knowledge_points = self._get_knowledge_points()
        
        metrics.update({
            "knowledge_points": knowledge_points,
            "knowledge_point_count": len(knowledge_points),
            "geometry_concepts": self._detect_geometry_concepts(),
            "problem_type": self._classify_problem_type(),
            "requires_theorem": self._requires_theorem(),
            "question_language": "zh",
            "has_multiple_choice": True,  # GeoQA总是多选题
            "choice_count": 4,  # 总是4个选项
            "complexity": self._assess_complexity()
        })
        
        return metrics
    
    def _assess_complexity(self) -> str:
        """评估问题复杂度"""
        if not self.question:
            return "unknown"
        
        complexity_score = 0
        
        # 知识点数量
        knowledge_points = self._get_knowledge_points()
        complexity_score += len(knowledge_points)
        
        # 几何概念数量
        concepts = self._detect_geometry_concepts()
        if len(concepts) > 2:
            complexity_score += 2
        elif len(concepts) > 1:
            complexity_score += 1
        
        # 问题类型
        problem_type = self._classify_problem_type()
        if problem_type in ["proof", "area_calculation"]:
            complexity_score += 2
        elif problem_type in ["angle_calculation", "length_calculation"]:
            complexity_score += 1
        
        # 需要定理
        if self._requires_theorem():
            complexity_score += 1
        
        # 分类
        if complexity_score >= 5:
            return "high"
        elif complexity_score >= 3:
            return "medium"
        else:
            return "low"