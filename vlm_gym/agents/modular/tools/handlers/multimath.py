# tools/handlers/multimath.py
from ..base import BaseTool, ToolConfig, ToolResult
from typing import Dict, Any, Tuple

class MultiMathTool(BaseTool):
    """MultiMath Server工具处理器"""
    
    def __init__(self, config: ToolConfig):
        super().__init__(config)
        self.supported_types = ['geometry', 'algebra', 'calculus']
        
    def can_handle(self, observation: Dict[str, Any]) -> Tuple[bool, float]:
        """判断是否能处理该任务"""
        question = observation.get("question", "").lower()
        
        # 几何问题关键词检测
        geometry_keywords = [
            'triangle', 'angle', 'circle', 'parallel', 'perpendicular',
            'congruent', 'similar', 'prove', 'theorem'
        ]
        
        # 检查是否包含几何关键词
        keyword_count = sum(1 for kw in geometry_keywords if kw in question)
        
        if keyword_count == 0:
            return False, 0.0
            
        # 计算置信度（基于关键词数量）
        confidence = min(keyword_count * 0.3, 1.0)
        
        # 如果明确标记为几何任务，提高置信度
        if observation.get("is_geometry_task"):
            confidence = max(confidence, 0.9)
            
        # 如果有CDL数据，大幅提高置信度
        if observation.get("cdl_data"):
            confidence = 0.95
            
        return True, confidence
    
    def build_prompt(self, observation: Dict[str, Any]) -> str:
        """构建工具调用提示"""
        question = observation.get("question", "")
        cdl_data = observation.get("cdl_data")
        
        tool_call = {
            "tool": "multimath_server",
            "parameters": {
                "task": "solve",
                "question": question,
                "problem_type": "geometry"
            }
        }
        
        # 如果有CDL数据，添加到参数中
        if cdl_data:
            tool_call["parameters"]["cdl_data"] = cdl_data
            tool_call["parameters"]["use_cdl"] = True
            
        return f'<tool_call>{json.dumps(tool_call)}</tool_call>'
    
    def process_result(self, raw_result: Any, observation: Dict[str, Any]) -> ToolResult:
        """处理工具返回的原始结果"""
        if isinstance(raw_result, dict):
            success = raw_result.get("success", False)
            answer = raw_result.get("answer", "")
            steps = raw_result.get("steps", [])
            confidence = raw_result.get("confidence", 0)
            
            return ToolResult(
                success=success,
                data={
                    "answer": answer,
                    "steps": steps,
                    "confidence": confidence
                },
                metadata={
                    "method": raw_result.get("method", "unknown"),
                    "time_taken": raw_result.get("time_taken", 0)
                }
            )
        else:
            return ToolResult(
                success=False,
                data=None,
                error="Invalid result format"
            )
    
    def format_for_answer(self, result: ToolResult, observation: Dict[str, Any]) -> str:
        """将工具结果格式化为最终答案的提示"""
        if not result.success:
            return f"MultiMath Server failed: {result.error}"
            
        data = result.data
        answer = data.get("answer", "")
        steps = data.get("steps", [])
        confidence = data.get("confidence", 0)
        
        prompt = f"""Based on MultiMath Server analysis:

Answer: {answer}
Confidence: {confidence:.1%}

"""
        if steps:
            prompt += "Solution steps:\n"
            for i, step in enumerate(steps[:5], 1):
                prompt += f"{i}. {step}\n"
                
        prompt += f"\nFinal answer: <answer>{answer}</answer>"
        
        return prompt