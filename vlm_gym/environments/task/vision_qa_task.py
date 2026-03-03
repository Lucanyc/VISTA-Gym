
# vlm_gym/environments/task/vision_qa_task.py

from typing import Tuple, Dict, Any, List, Optional
import json
from pathlib import Path
from PIL import Image

from .base import BaseTask


class VisionQATask(BaseTask):
    """视觉问答任务
    
    处理各种视觉问答数据集，如ChartQA、ScienceQA、VizWiz等
    """
    
    def __init__(self, task_id: str, adapter, **kwargs):
        """
        初始化视觉问答任务
        
        Args:
            task_id: 任务ID
            adapter: 数据适配器
            **kwargs: 额外参数
        """
        super().__init__(task_id, adapter, **kwargs)
        
        # 任务特定属性
        self.image_path: Optional[str] = None
        self.question: Optional[str] = None
        self.choices: Optional[List[str]] = None
        self.answer: Optional[str] = None
        self.answer_index: Optional[int] = None
        self.metadata: Dict[str, Any] = {}
        
    def setup(self) -> Tuple[str, Dict[str, Any]]:
        """
        设置任务
        
        Returns:
            task_goal: 任务目标
            task_info: 任务信息
        """
        # 从task_data中提取信息
        #self.image_path = self.task_data.get("image_path")
        #self.question = self.task_data.get("question", "")
        #self.choices = self.task_data.get("choices", [])
        #self.answer = self.task_data.get("answer")
        #self.answer_index = self.task_data.get("answer_index")
        #self.metadata = self.task_data.get("metadata", {})
        
        self.image_path = self.task_data.get("image_path")  # 可以为 None
        self.question = self.task_data.get("question") or ""  # 确保是字符串
        self.choices = self.task_data.get("choices") or []  # 确保是列表
        self.answer = self.task_data.get("answer")  # 可以为 None
        self.answer_index = self.task_data.get("answer_index")  # 可以为 None
        self.metadata = self.task_data.get("metadata") or {}  # 确保是字典
        
        
        # 构建任务目标
        task_goal = self._build_task_goal()
        
        # 构建任务信息
        task_info = {
            "question_type": self.metadata.get("question_type", "general"),
            "difficulty": self.metadata.get("difficulty", "medium"),
            "dataset": self.metadata.get("dataset", "unknown"),
            "has_choices": len(self.choices) > 0,
            "num_choices": len(self.choices),
            "image_available": self.image_path is not None
        }
        
        self._setup_complete = True
        return task_goal, task_info
    
    def _build_task_goal(self) -> str:
        """构建任务目标描述"""
        goal_parts = []
        
        # 基本任务描述
        goal_parts.append("Please analyze the provided image and answer the following question:")
        goal_parts.append(f"\nQuestion: {self.question}")
        
        # 如果有选择题选项
        if self.choices:
            goal_parts.append("\nChoices:")
            for i, choice in enumerate(self.choices):
                goal_parts.append(f"  {chr(65+i)}. {choice}")
            goal_parts.append("\nPlease select the correct answer from the choices above.")
        else:
            goal_parts.append("\nPlease provide a detailed answer to the question.")
        
        return "\n".join(goal_parts)
    
    def check_success(self, action: Any) -> Tuple[bool, str]:
        """
        检查动作是否成功
        
        Args:
            action: 用户的答案
            
        Returns:
            success: 是否成功
            feedback: 反馈信息
        """
        if not action:
            return False, "No answer provided"
        
        # 处理选择题
        if self.choices and self.answer_index is not None:
            # 检查字母答案 (A, B, C, D...)
            if isinstance(action, str) and len(action) == 1 and action.upper().isalpha():
                selected_index = ord(action.upper()) - ord('A')
                if selected_index == self.answer_index:
                    return True, "Correct! Well done."
                else:
                    correct_letter = chr(65 + self.answer_index)
                    return False, f"Incorrect. The correct answer is {correct_letter}."
            
            # 检查索引答案
            try:
                selected_index = int(action)
                if selected_index == self.answer_index:
                    return True, "Correct! Well done."
                else:
                    return False, f"Incorrect. The correct answer is option {self.answer_index}."
            except ValueError:
                pass
        
        # 处理开放式问答
        if self.answer:
            # 简单的文本匹配（可以根据需要改进）
            if str(action).lower().strip() == str(self.answer).lower().strip():
                return True, "Correct! Your answer matches the expected answer."
            else:
                # 部分匹配检查
                if str(self.answer).lower() in str(action).lower():
                    return True, "Correct! Your answer contains the key information."
                else:
                    return False, f"Incorrect. Expected: {self.answer}"
        
        # 没有标准答案的情况
        if action and len(str(action)) > 10:
            return True, "Answer received. Manual evaluation may be needed."
        else:
            return False, "Please provide a more detailed answer."
    
    def validate(
        self,
        chat_history: List[Dict],
        observation: Any,
        full_history: Optional[List[Any]] = None
    ) -> Tuple[float, bool, str, Dict[str, Any]]:
        """
        验证任务执行情况
        
        Args:
            chat_history: 聊天历史
            observation: 当前观察
            full_history: 完整历史
            
        Returns:
            reward: 奖励值
            done: 是否完成
            message: 反馈消息
            info: 额外信息
        """
        # 从观察中提取答案
        answer = None
        if hasattr(observation, 'content'):
            answer = observation.content
        elif isinstance(observation, dict):
            answer = observation.get('answer') or observation.get('content')
        elif isinstance(observation, str):
            answer = observation
        
        # 检查成功
        success, feedback = self.check_success(answer)
        
        # 计算奖励
        reward = 1.0 if success else 0.0
        
        # 构建信息
        info = {
            "success": success,
            "feedback": feedback,
            "task_id": self.task_id,
            "steps": len(full_history) if full_history else len(chat_history),
            "answer_provided": answer,
            "correct_answer": self.answer or (self.choices[self.answer_index] if self.answer_index is not None and self.choices else None)
        }
        
        # 任务完成
        done = True
        
        return reward, done, feedback, info
    
    def get_observation(self) -> Dict[str, Any]:
        """获取当前观察"""
        obs = super().get_observation()
        
        # 添加VQA特定信息
        obs["vqa_info"] = {
            "image_path": self.image_path,
            "question": self.question,
            "choices": self.choices,
            "has_standard_answer": self.answer is not None or self.answer_index is not None
        }
        
        return obs
    
    def get_info(self) -> Dict[str, Any]:
        """获取任务信息"""
        info = super().get_info()
        info.update({
            "question": self.question,
            "num_choices": len(self.choices) if self.choices else 0,
            "question_type": self.metadata.get("question_type", "unknown"),
            "dataset": self.metadata.get("dataset", "unknown")
        })
        return info
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取任务指标"""
        return {
            "task_id": self.task_id,
            "question_length": len(self.question) if self.question else 0,
            "has_image": self.image_path is not None,
            "is_multiple_choice": len(self.choices) > 0 if self.choices else False,
            "metadata": self.metadata
        }
    
    def teardown(self) -> None:
        """清理资源"""
        self.image_path = None
        self.question = None
        self.choices = None
        self.answer = None
        self.answer_index = None
        self.metadata = {}
        self._setup_complete = False