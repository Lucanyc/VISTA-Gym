# strategies/workflow_manager.py
from typing import Dict, Any, List, Optional, Tuple
from ..tools.coordinators.base_coordinator import BaseCoordinator
import logging

class WorkflowManager:
    """工作流管理器 - 选择和执行合适的工作流"""
    
    def __init__(self):
        self.coordinators: Dict[str, BaseCoordinator] = {}
        self.logger = logging.getLogger(__name__)
        
    def register_coordinator(self, coordinator: BaseCoordinator):
        """注册工作流协调器"""
        self.coordinators[coordinator.name] = coordinator
        self.logger.info(f"Registered coordinator: {coordinator.name}")
        
    def select_workflow(self, observation: Dict[str, Any]) -> Optional[BaseCoordinator]:
        """为任务选择最合适的工作流"""
        candidates = []
        
        for coordinator in self.coordinators.values():
            can_handle, confidence = coordinator.can_handle(observation)
            if can_handle:
                candidates.append((coordinator, confidence))
                
        if not candidates:
            return None
            
        # 选择置信度最高的工作流
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = candidates[0][0]
        
        self.logger.info(f"Selected workflow: {selected.name} (confidence: {candidates[0][1]:.2f})")
        return selected
    
    def execute_workflow(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """执行工作流"""
        coordinator = self.select_workflow(observation)
        
        if not coordinator:
            # 没有合适的工作流，返回错误
            return "No suitable workflow found", {"error": True}
            
        return coordinator.execute(observation)