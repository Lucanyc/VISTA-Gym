
from typing import Dict, List, Tuple, Any
import re
from .base import AbstractActionSet

class VLMActionSet(AbstractActionSet):
    """VLM action set"""
    
    def __init__(self):
        super().__init__()
        
        # define actions 
        self.actions = {
            "analyze_image": {
                "description": "Analyze an image and extract information",
                "params": ["type"],
                "examples": [
                    'analyze_image(type="objects")',
                    'analyze_image(type="scene")',
                    'analyze_image(type="text")'
                ]
            },
            "answer_question": {
                "description": "Answer a question about the image",
                "params": ["question", "context"],
                "examples": [
                    'answer_question(question="What is in the image?")',
                    'answer_question(question="How many people?", context={"focus": "crowd"})'
                ]
            },
            "describe_region": {
                "description": "Describe a specific region in the image",
                "params": ["bbox"],
                "examples": [
                    'describe_region(bbox=[100, 100, 200, 200])',
                    'describe_region(bbox=[0, 0, 50, 50])'
                ]
            },
            "compare_images": {
                "description": "Compare multiple images",
                "params": ["image_ids", "aspect"],
                "examples": [
                    'compare_images(image_ids=["img1", "img2"], aspect="similarity")',
                    'compare_images(image_ids=["img1", "img2", "img3"], aspect="differences")'
                ]
            }
        }
    
    def parse_action(self, action_str: str) -> Tuple[str, Dict[str, Any]]:
        """parse action string
        
        Args:
            action_str: action string，such as 'analyze_image(type="objects")'
            
        Returns:
            (action_type, params): 
        """
       
        match = re.match(r'(\w+)\((.*)\)', action_str.strip())
        if not match:
            raise ValueError(f"Invalid action format: {action_str}")
            
        action_type = match.group(1)
        params_str = match.group(2)
        
        if action_type not in self.actions:
            raise ValueError(f"Unknown action: {action_type}")
        
   
        params = {}
        if params_str:
       
            param_pairs = re.findall(r'(\w+)=([^,]+)', params_str)
            for key, value in param_pairs:
         
                value = value.strip().strip('"\'')
        
                if value.startswith('[') and value.endswith(']'):
                    value = eval(value) 
                params[key] = value
                
        return action_type, params
    
    def describe(self, with_examples: bool = True) -> str:
    
        description = f"VLM Action Space ({len(self.actions)} actions available):\n\n"
        
        for action_name, action_info in self.actions.items():
            description += f"- {action_name}: {action_info['description']}\n"
            description += f"  Parameters: {', '.join(action_info['params'])}\n"
            
            if with_examples and action_info['examples']:
                description += "  Examples:\n"
                for example in action_info['examples']:
                    description += f"    {example}\n"
            description += "\n"
            
        return description
    
    def validate_action(self, action_str: str) -> bool:

        try:
            action_type, params = self.parse_action(action_str)
            return True
        except:
            return False
