# tools/coordinators/geometry_workflow.py
from .base_coordinator import BaseCoordinator, WorkflowContext, WorkflowState
from typing import Dict, Any, List, Tuple

class GeometryWorkflowCoordinator(BaseCoordinator):
    """几何问题工作流协调器 - DiagramFormalizer + MultiMath"""
    
    def __init__(self):
        super().__init__("geometry_workflow")
        
    def can_handle(self, observation: Dict[str, Any]) -> Tuple[bool, float]:
        """判断是否能处理该任务"""
        question = observation.get("question", "").lower()
        
        # 几何问题特征
        geometry_indicators = [
            'prove', 'theorem', 'congruent', 'similar',
            'angle', 'triangle', 'circle', 'parallel'
        ]
        
        # 需要形式化的指标
        formalization_indicators = [
            'prove that', 'show that', 'verify',
            'given that', 'if...then'
        ]
        
        has_geometry = any(ind in question for ind in geometry_indicators)
        needs_formalization = any(ind in question for ind in formalization_indicators)
        
        if not has_geometry:
            return False, 0.0
            
        # 如果需要形式化，置信度更高
        if needs_formalization:
            return True, 0.9
        else:
            return True, 0.6
    
    def define_workflow(self, context: WorkflowContext) -> List[str]:
        """定义工作流步骤"""
        steps = []
        
        # 第一步：总是先用DiagramFormalizer提取CDL
        steps.append("diagram_formalizer")
        
        # 第二步：根据CDL结果决定是否需要MultiMath
        # （这个决定会在prepare_tool_input中做）
        if context.shared_data.get("needs_calculation", True):
            steps.append("multimath_server")
            
        return steps
    
    def prepare_tool_input(self, tool_name: str, context: WorkflowContext) -> Dict[str, Any]:
        """为特定工具准备输入"""
        base_input = context.observation.copy()
        
        if tool_name == "diagram_formalizer":
            # DiagramFormalizer的输入就是原始观察
            return base_input
            
        elif tool_name == "multimath_server":
            # 为MultiMath准备输入，包含CDL数据
            cdl_result = context.tool_results.get("diagram_formalizer", {})
            
            if cdl_result and cdl_result.get("success"):
                # 提取CDL数据
                cdl_data = cdl_result.get("cdl_data", {})
                base_input["cdl_data"] = cdl_data
                
                # 如果有形式化的问题陈述，使用它
                if cdl_data.get("formalized_problem"):
                    base_input["question"] = cdl_data["formalized_problem"]
                    
                # 添加约束和已知条件
                if cdl_data.get("constraints"):
                    base_input["constraints"] = cdl_data["constraints"]
                    
                if cdl_data.get("known_values"):
                    base_input["known_values"] = cdl_data["known_values"]
                    
            return base_input
            
        return base_input
    
    def should_continue(self, context: WorkflowContext) -> bool:
        """判断是否继续执行工作流"""
        # 如果已经失败，不继续
        if context.state == WorkflowState.FAILED:
            return False
            
        # 如果DiagramFormalizer失败，但不是关键失败，可以继续
        if context.current_step == 1:  # 准备执行MultiMath
            df_result = context.tool_results.get("diagram_formalizer", {})
            if not df_result.get("success"):
                # 如果形式化失败但不严重，仍然尝试MultiMath
                if df_result.get("partial_success"):
                    return True
                # 完全失败则停止
                return False
                
        return True
    
    def generate_final_answer(self, context: WorkflowContext) -> Tuple[str, Dict[str, Any]]:
        """生成最终答案"""
        # 获取工具结果
        df_result = context.tool_results.get("diagram_formalizer", {})
        mm_result = context.tool_results.get("multimath_server", {})
        
        # 优先使用MultiMath的答案
        if mm_result and mm_result.get("success"):
            answer = mm_result.get("answer", "")
            confidence = mm_result.get("confidence", 0)
            
            prompt = f"""Based on formal geometric analysis:

1. Diagram Formalization: {df_result.get("status", "completed")}
2. Mathematical Solution: {mm_result.get("method", "analytical")}

Answer: {answer}
Confidence: {confidence:.1%}

<answer>{answer}</answer>"""
            
            return prompt, {
                "workflow": "geometry",
                "tools_used": ["diagram_formalizer", "multimath_server"],
                "success": True
            }
            
        # 如果只有DiagramFormalizer的结果
        elif df_result and df_result.get("success"):
            analysis = df_result.get("analysis", "")
            
            prompt = f"""Based on diagram formalization:

{analysis}

<answer>{df_result.get("conclusion", "Unable to determine")}</answer>"""
            
            return prompt, {
                "workflow": "geometry",
                "tools_used": ["diagram_formalizer"],
                "partial_success": True
            }
            
        # 两个工具都失败
        return self.handle_error(context)