from typing import Tuple, Dict, Any, List, Optional
import json
from pathlib import Path
from PIL import Image
import numpy as np

from .base import AbstractVLMTask

class VisionQATask(AbstractVLMTask):
    """vision qa task"""
    
    def __init__(
        self,
        task_id: str,
        dataset_path: str,
        **kwargs
    ):
        super().__init__(task_id)
        self.dataset_path = Path(dataset_path)
        self.current_image: Optional[Image.Image] = None
        self.current_question: Optional[str] = None
        self.expected_answer: Optional[str] = None
        self.task_data: Dict[str, Any] = {}
        
    def setup(self) -> Tuple[str, Dict[str, Any]]:
        """task setting"""
        
        task_file = self.dataset_path / f"{self.task_id}.json"
        if not task_file.exists():
            raise ValueError(f"Task file not found: {task_file}")
            
        with open(task_file, 'r') as f:
            self.task_data = json.load(f)
            
        # load image
        image_path = self.dataset_path / self.task_data["image"]
        self.current_image = Image.open(image_path)
        
        # set questions and answers
        self.current_question = self.task_data["question"]
        self.expected_answer = self.task_data.get("answer")
        
        # build task goal
        task_goal = f"Please answer the following question about the image: {self.current_question}"
        
        task_info = {
            "image_size": self.current_image.size,
            "question_type": self.task_data.get("question_type", "general"),
            "difficulty": self.task_data.get("difficulty", "medium"),
        }
        
        return task_goal, task_info
    
    def validate(
        self,
        chat_history: List[Dict],
        last_observation: Any,
        full_history: List[Any]
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """verify the task"""
        reward = 0.0
        done = False
        info = {}
        
        # 检查是否有答案
        if last_observation.type == "answer":
            user_answer = last_observation.content
            
            # calculate reward (simple design for now)
            if self.expected_answer:
                # simple text matching
                if user_answer.lower() == self.expected_answer.lower():
                    reward = 1.0
                    info["correct"] = True
                else:
                    reward = 0.0
                    info["correct"] = False
                    
                done = True
            else:
                # if no answer, give other reward
                reward = 0.5  # reward for part answer
                done = True
                info["evaluated"] = "no_ground_truth"
        
        # record other info
        info["steps"] = len(full_history)
        info["task_id"] = self.task_id
        
        return reward, done, info
    
    def get_info(self) -> Dict[str, Any]:
        """get task info""
        return {
            "task_id": self.task_id,
            "question": self.current_question,
            "has_image": self.current_image is not None,
            "metadata": self.task_data.get("metadata", {})
        }
    
    def teardown(self):
        """clear up the resource"""
        self.current_image = None
        self.current_question = None
        self.expected_answer = None
        self.task_data = {}
