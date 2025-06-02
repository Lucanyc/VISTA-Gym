"""VQA task in LLaVA format"""

from pathlib import Path
from typing import Tuple, Dict, Any, List
from PIL import Image


class LLaVATask:
    """Vision‑Language Question‑Answering task in the LLaVA format"""

    def __init__(self, task_id: str, adapter, **kwargs):
        self.task_id = task_id
        self.adapter = adapter
        self.task_data = None
        self.current_image = None
        self.current_question = None  # Store the current question

    def setup(self) -> Tuple[str, Dict[str, Any]]:
        """Initialise the task and return (task_goal, task_info)."""
        # Retrieve task metadata from the adapter
        self.task_data = self.adapter.get_task_data(self.task_id)

        # Load the image
        image_path = self.task_data["image_path"]
        if Path(image_path).exists():
            self.current_image = Image.open(image_path)
        else:
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Store the current question
        self.current_question = self.task_data["question"]

        # In this task, the goal is simply to answer the question
        task_goal = self.current_question

        task_info = {
            "task_id": self.task_id,
            "dataset_type": self.task_data["dataset_type"],
            "image_size": self.current_image.size if self.current_image else None,
            "image_path": image_path,  # Pass the absolute path back to the environment
            "has_ground_truth": bool(self.task_data.get("answer")),
            "metadata": self.task_data.get("metadata", {})
        }

        return task_goal, task_info

    def validate(
        self,
        chat_history: List[Dict],
        last_observation: Any,
        full_history: List[Any]
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """Check whether the task has been solved and assign a reward."""
        reward = 0.0
        done = False
        info: Dict[str, Any] = {}

        # Ensure the last observation is an answer produced by the agent
        if hasattr(last_observation, "type") and last_observation.type == "answer":
            user_answer = str(last_observation.content)
            ground_truth = self.task_data.get("answer", "")

            # Evaluate the answer against ground truth (if available)
            if ground_truth:
                user_answer_lower = user_answer.lower()
                ground_truth_lower = ground_truth.lower()

                # Use dataset‑specific evaluation where appropriate
                if self.task_data["dataset_type"] == "chartqa":
                    reward = self._evaluate_chartqa_answer(user_answer_lower, ground_truth_lower)
                else:
                    reward = self._evaluate_general_answer(user_answer_lower, ground_truth_lower)

                info["correct"] = reward > 0.5
            else:
                # When no reference answer is provided, give partial credit for substantial replies
                if len(user_answer) > 20:
                    reward = 0.6

            done = True
            info.update(
                {
                    "ground_truth": ground_truth,
                    "user_answer": user_answer,
                    "reward": reward,
                    "answer_length": len(user_answer),
                }
            )

        return reward, done, info

    def _evaluate_chartqa_answer(self, user_answer: str, ground_truth: str) -> float:
        """Scoring logic tailored for ChartQA answers."""
        import re

        def extract_numbers(text: str):
            return re.findall(r"\d+\.?\d*", text)

        user_numbers = extract_numbers(user_answer)
        truth_numbers = extract_numbers(ground_truth)

        # Exact numeric match grants full credit
        if truth_numbers and user_numbers:
            for truth_num in truth_numbers:
                if truth_num in user_numbers:
                    return 1.0

        # Look for trend‑related keywords
        trend_words = ["increasing", "decreasing", "stable", "trend", "rise", "fall"]
        for word in trend_words:
            if word in ground_truth and word in user_answer:
                return 0.8

        # Partial lexical overlap yields partial credit
        if any(word in user_answer for word in ground_truth.split()):
            return 0.5

        return 0.0

    def _evaluate_general_answer(self, user_answer: str, ground_truth: str) -> float:
        """Generic answer‑matching heuristic used for non‑ChartQA datasets."""
        # Full containment gets full credit
        if ground_truth in user_answer:
            return 1.0

        # Partial keyword overlap gets moderate credit
        ground_truth_words = set(ground_truth.split())
        user_answer_words = set(user_answer.split())

        common_words = ground_truth_words.intersection(user_answer_words)
        if len(common_words) > len(ground_truth_words) * 0.5:
            return 0.7

        return 0.0

    def get_info(self) -> Dict[str, Any]:
        """Return task metadata for logging or debugging."""
        return {
            "task_id": self.task_id,
            "question": self.current_question,
            "dataset_type": self.task_data.get("dataset_type", "unknown"),
            "has_answer": bool(self.task_data.get("answer")),
            "image_path": self.task_data.get("image_path", ""),
        }

    def teardown(self):
        """Release any open resources (e.g., image files)."""
        if self.current_image:
            self.current_image.close()
        self.current_image = None
        self.current_question = None
        self.task_data = None
