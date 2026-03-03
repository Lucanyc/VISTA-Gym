# vlm_gym/environments/strategies/spatial_reasoning.py

class SpatialReasoningStrategy:
    """空间推理任务的教学策略"""
    
    def __init__(self):
        self.reasoning_steps = [
            "identify_objects",
            "analyze_positions",
            "determine_relationships",
            "apply_spatial_rules",
            "verify_conclusion"
        ]
    
    def generate_prompts(self, current_step):
        prompts = {
            "identify_objects": "First, can you identify all the objects in the image?",
            "analyze_positions": "Now, describe the position of each object.",
            "determine_relationships": "What are the spatial relationships between these objects?",
            # ...
        }
        return prompts[current_step]