##adaptor for both scienceqa and chartqa datasets

import json
from pathlib import Path
from typing import Dict, List, Any, Optional

class UnifiedVQAAdapter:
    
    
    def __init__(self, data_root: str, annotation_file: str):
        self.data_root = Path(data_root)
        self.annotation_file = Path(annotation_file)
        

        self.annotations = self._load_annotations()
        

        self.id_to_data = {item["id"]: item for item in self.annotations}
        

        self.chartqa_items = []
        self.scienceqa_items = []
        self._categorize_items()
        
    def _load_annotations(self) -> List[Dict[str, Any]]:

        with open(self.annotation_file, 'r') as f:
            data = json.load(f)
            
        if isinstance(data, list):
            return data
        else:
            raise ValueError(f"Expected list format in {self.annotation_file}")
    
    def _categorize_items(self):

        for item in self.annotations:
            if item.get("images"):
                image_path = item["images"][0]
                if "chartqa" in image_path:
                    self.chartqa_items.append(item)
                elif "scienceqa" in image_path:
                    self.scienceqa_items.append(item)
    
    def get_task_ids(self, dataset_filter: Optional[str] = None) -> List[str]:

        if dataset_filter == "chartqa":
            return [item["id"] for item in self.chartqa_items]
        elif dataset_filter == "scienceqa":
            return [item["id"] for item in self.scienceqa_items]
        else:
            return [item["id"] for item in self.annotations]
    
    def get_task_data(self, task_id: str) -> Dict[str, Any]:

        if task_id not in self.id_to_data:
            raise ValueError(f"Task {task_id} not found")
            
        item = self.id_to_data[task_id]
        

        conversations = item.get("conversations", [])
        question = ""
        answer = ""
        
        for conv in conversations:
            if conv["from"] == "human":
                # remove <image> label
                question = conv["value"].replace("<image>", "").strip()
            elif conv["from"] == "gpt":
                answer = conv["value"]
        
        # get path for image
        images = item.get("images", [])
        image_path = ""
        dataset_type = "unknown"
        
        if images:

            rel_image_path = images[0]
            image_path = str(self.data_root / rel_image_path)
            

            if "chartqa" in rel_image_path:
                dataset_type = "chartqa"
            elif "scienceqa" in rel_image_path:
                dataset_type = "scienceqa"
        
        return {
            "task_id": task_id,
            "image_path": image_path,
            "question": question,
            "answer": answer,
            "dataset_type": dataset_type,
            "conversations": conversations,
            "metadata": {
                "original_id": item.get("id"),
                "num_turns": len(conversations) // 2
            }
        }
    
    def get_dataset_stats(self) -> Dict[str, Any]:

        stats = {
            "total_samples": len(self.annotations),
            "chartqa_samples": len(self.chartqa_items),
            "scienceqa_samples": len(self.scienceqa_items),
            "questions": {
                "unique_questions": len(set(self.get_task_data(item["id"])["question"] 
                                          for item in self.annotations)),
                "avg_question_length": 0
            }
        }
        

        total_length = sum(len(self.get_task_data(item["id"])["question"]) 
                          for item in self.annotations)
        stats["questions"]["avg_question_length"] = total_length / max(1, len(self.annotations))
        
        return stats
