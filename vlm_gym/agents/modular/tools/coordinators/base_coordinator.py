# tools/coordinators/base_coordinator.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

class WorkflowState(Enum):
    """工作流状态"""
    INIT = "initialized"
    RUNNING = "running"
    WAITING = "waiting_for_tool"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retrying"

@dataclass
class WorkflowContext:
    """工作流上下文 - 存储工作流执行过程中的所有数据"""
    observation: Dict[str, Any]
    state: WorkflowState = WorkflowState.INIT
    current_step: int = 0
    tool_results: Dict[str, Any] = field(default_factory=dict)
    shared_data: Dict[str, Any] = field(default_factory=dict)  # 工具间共享数据
    history: List[Dict] = field(default_factory=list)
    error: Optional[str] = None
    
    def add_tool_result(self, tool_name: str, result: Any):
        """添加工具执行结果"""
        self.tool_results[tool_name] = result
        self.history.append({
            "step": self.current_step,
            "tool": tool_name,
            "result": result
        })
        
    def get_last_result(self) -> Optional[Any]:
        """获取最后一个工具的结果"""
        if self.history:
            return self.history[-1]["result"]
        return None

class BaseCoordinator(ABC):
    """工作流协调器基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.registry = None  # 将由系统注入
        
    @abstractmethod
    def can_handle(self, observation: Dict[str, Any]) -> Tuple[bool, float]:
        """判断是否能处理该任务"""
        pass
    
    @abstractmethod
    def define_workflow(self, context: WorkflowContext) -> List[str]:
        """定义工作流步骤（返回工具名称列表）"""
        pass
    
    @abstractmethod
    def prepare_tool_input(self, tool_name: str, context: WorkflowContext) -> Dict[str, Any]:
        """为特定工具准备输入"""
        pass
    
    @abstractmethod
    def should_continue(self, context: WorkflowContext) -> bool:
        """判断是否继续执行工作流"""
        pass
    
    def execute(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """执行工作流"""
        context = WorkflowContext(observation=observation)
        context.state = WorkflowState.RUNNING
        
        try:
            # 定义工作流步骤
            workflow_steps = self.define_workflow(context)
            self.logger.info(f"Workflow defined: {workflow_steps}")
            
            # 执行每个步骤
            for i, tool_name in enumerate(workflow_steps):
                context.current_step = i
                
                # 检查是否应该继续
                if not self.should_continue(context):
                    self.logger.info(f"Workflow stopped at step {i}")
                    break
                
                # 准备工具输入
                tool_input = self.prepare_tool_input(tool_name, context)
                
                # 获取并执行工具
                tool = self.registry.get_tool(tool_name)
                if not tool:
                    raise ValueError(f"Tool not found: {tool_name}")
                
                # 构建提示并返回（实际执行会在外部进行）
                context.state = WorkflowState.WAITING
                prompt = tool.build_prompt(tool_input)
                
                # 这里返回工具调用，等待外部执行
                return prompt, {"workflow_context": context, "waiting_for": tool_name}
            
            # 工作流完成
            context.state = WorkflowState.COMPLETED
            return self.generate_final_answer(context)
            
        except Exception as e:
            context.state = WorkflowState.FAILED
            context.error = str(e)
            self.logger.error(f"Workflow failed: {e}")
            return self.handle_error(context)
    
    def resume_after_tool(self, context: WorkflowContext, tool_result: Any) -> Tuple[str, Dict[str, Any]]:
        """工具执行后恢复工作流"""
        # 存储工具结果
        current_tool = context.history[-1]["tool"] if context.history else "unknown"
        context.add_tool_result(current_tool, tool_result)
        
        # 继续执行工作流
        return self.execute(context.observation)
    
    @abstractmethod
    def generate_final_answer(self, context: WorkflowContext) -> Tuple[str, Dict[str, Any]]:
        """生成最终答案"""
        pass
    
    def handle_error(self, context: WorkflowContext) -> Tuple[str, Dict[str, Any]]:
        """处理错误"""
        return f"Workflow failed: {context.error}", {"error": True}
