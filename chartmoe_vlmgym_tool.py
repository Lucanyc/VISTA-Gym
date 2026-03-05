# chartmoe_vlmgym_tool.py
"""
ChartMoE VLMGym Tool
"""
from PIL import Image
import json
import os
import logging
from pathlib import Path
import torch
from torchvision import transforms

# Set environment variables
os.environ["HF_TORCH_LOAD_DISABLE_SAFE_CHECK"] = "1"
os.environ["TRUST_REMOTE_CODE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# Apply all fixes
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None

original_load = torch.load
torch.load = lambda f, *args, **kwargs: original_load(f, *args, weights_only=False, **kwargs)

import transformers.modeling_utils
import safetensors.torch

def patched_load_state_dict(checkpoint_file, map_location="cpu", **kwargs):
    if map_location == "meta":
        map_location = "cpu"
    if hasattr(checkpoint_file, 'endswith') and checkpoint_file.endswith('.safetensors'):
        return safetensors.torch.load_file(checkpoint_file, device=str(map_location))
    return original_load(checkpoint_file, map_location=map_location, weights_only=False)

transformers.modeling_utils.load_state_dict = patched_load_state_dict

# Import required modules
try:
    from vlm_gym.environments.tools.chart.chartmoe import ChartMoETool
except ImportError:
    # If vlm_gym is not installed, create a base class
    class ChartMoETool:
        pass

from transformers import AutoModel, AutoTokenizer

# Global variables for model storage (singleton pattern)
_model_instances = {}
_tokenizer_instances = {}

def find_chartmoe_model():
    """Automatically find ChartMoE model path"""
    possible_paths = [
        "/workspace/mathvista/model",
        "/workspace/model",
        "/workspace/chartmoe",
        "/workspace/models/chartmoe",
        "/models/chartmoe",
    ]
    
    for path in possible_paths:
        model_path = Path(path)
        if model_path.exists() and model_path.is_dir():
            # Check if it contains required model files
            if (model_path / "config.json").exists():
                print(f"Found ChartMoE model: {path}")
                return str(model_path.absolute())
    
    return None

def get_model_and_tokenizer(config):
    """Get or create model instance (singleton)"""
    global _model_instances, _tokenizer_instances
    
    # Get model identifier
    model_identifier = config.get('model_path') or config.get('model_name')
    
    # If no path specified, try to find automatically
    if not model_identifier:
        found_path = find_chartmoe_model()
        if found_path:
            model_identifier = found_path
            print(f"Using auto-discovered model path: {model_identifier}")
        else:
            model_identifier = "/workspace/mathvista/model"
            print(f"Using default model path: {model_identifier}")
    
    # Check if it is a local path
    model_path = Path(model_identifier)
    is_local = model_path.exists() and model_path.is_dir()
    
    # Use model path as key
    cache_key = str(model_path.absolute()) if is_local else model_identifier
    
    if cache_key not in _model_instances:
        print(f"Loading ChartMoE model for the first time: {model_identifier}")
        
        # Determine device
        device = config.get('device', 'cuda')
        if device == 'cpu':
            device_map = 'cpu'
        else:
            device_map = device if ':' in device else f"{device}:0"
        
        # Prepare loading arguments
        load_kwargs = {
            "trust_remote_code": config.get('trust_remote_code', True),
            "torch_dtype": torch.float16 if device != 'cpu' else torch.float32,
            "device_map": device_map,
        }
        
        # Load local model
        if is_local:
            model_path_str = str(model_path.absolute())
            print(f"Detected local model path: {model_path_str}")
            
            _model_instances[cache_key] = AutoModel.from_pretrained(
                model_path_str,
                local_files_only=True,
                **load_kwargs
            )
            
            _tokenizer_instances[cache_key] = AutoTokenizer.from_pretrained(
                model_path_str,
                trust_remote_code=config.get('trust_remote_code', True),
                local_files_only=True
            )
        else:
            raise ValueError(f"ChartMoE model not found at path: {model_identifier}")
        
        print("Model loaded successfully")
    
    return _model_instances[cache_key], _tokenizer_instances[cache_key]

class FixedChartMoETool(ChartMoETool):
    def __init__(self, config):
        # Initialize basic attributes
        self.config = config
        self.device = config.get('device', 'cuda')
        self.current_image = None
        self.model = None
        self.tokenizer = None
        
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Lazy model loading
        self._model_loaded = False
        
        # Get parameters from config
        self.max_new_tokens = config.get('max_new_tokens', 800)
        
        # ChartMoE uses 490x490 image size
        self.image_size = 490
    
    def _load_model(self):
        """Lazy load model"""
        if not self._model_loaded:
            try:
                self.model, self.tokenizer = get_model_and_tokenizer(self.config)
                self._model_loaded = True
                print(f"Model loaded successfully, using image size: {self.image_size}x{self.image_size}")
            except Exception as e:
                self.logger.error(f"Model loading failed: {str(e)}")
                raise
    
    def reset(self, image):
        """Reset tool with new image"""
        # Ensure image is a PIL Image
        if isinstance(image, str):
            image = Image.open(image)
        
        # Important: convert to RGB (remove alpha channel)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        self.current_image = image
        self.logger.info(f"Reset with image size: {image.size}, mode: {image.mode}")
    
    def execute(self, params_str):
        """Execute chart analysis task"""
        try:
            # Ensure model is loaded
            self._load_model()
            
            # Parse parameters
            params = json.loads(params_str) if isinstance(params_str, str) else params_str
            
            task = params.get('task', '')
            prompt = params.get('prompt', '')
            
            # Task mapping
            task_prompts = {
                'to_table': 'Convert this chart to a table format with clear rows and columns. Include all visible data.',
                'describe': 'Describe this chart in detail, including title, axes, data series, and key insights.',
                'extract_data': 'Extract all numerical data values and labels from this chart.',
                'summarize': 'Provide a brief summary of what this chart shows.',
                'analyze': 'Analyze this chart and provide key insights and patterns.',
                'compare': 'Compare the different data series or categories shown in this chart.',
                'trend': 'Identify and describe any trends visible in this chart.',
                'title': 'What is the title of this chart?'
            }
            
            query = task_prompts.get(task, prompt or "Analyze this chart.")
            formatted_query = f"<ImageHere>{query}"
            
            if self.current_image is None:
                return {"error": "No image loaded", "error_type": "NoImageError"}
            
            # Process image - use 490x490
            rgb_image = self.current_image.convert('RGB') if self.current_image.mode != 'RGB' else self.current_image
            
            # Create preprocessing pipeline - use 490x490
            preprocess = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073], 
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])
            
            # Apply preprocessing
            image_tensor = preprocess(rgb_image).unsqueeze(0).to(self.model.device)
            
            # Adjust data type based on device
            if self.device != 'cpu':
                image_tensor = image_tensor.half()
            
            self.logger.info(f"Processed image tensor shape: {image_tensor.shape}, device: {image_tensor.device}")
            
            # Call model
            try:
                response = self.model.chat(
                    tokenizer=self.tokenizer,
                    query=formatted_query,
                    image=image_tensor,
                    history=[],
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False
                )
                
                # Process response (chat returns tuple)
                if isinstance(response, tuple):
                    response = response[0]
                
                return {
                    "processed_output": response,
                    "task_type": task,
                    "success": True
                }
                
            except Exception as e:
                self.logger.error(f"Model inference failed: {str(e)}")
                # If failed, try with fewer max_new_tokens
                if "out of memory" in str(e).lower():
                    self.logger.info("Retrying with fewer tokens...")
                    response = self.model.chat(
                        tokenizer=self.tokenizer,
                        query=formatted_query,
                        image=image_tensor,
                        history=[],
                        max_new_tokens=200,
                        do_sample=False
                    )
                    
                    if isinstance(response, tuple):
                        response = response[0]
                    
                    return {
                        "processed_output": response,
                        "task_type": task,
                        "success": True,
                        "note": "Used reduced max_new_tokens"
                    }
                else:
                    raise
            
        except Exception as e:
            self.logger.error(f"ChartMoE execution failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "error_type": type(e).__name__,
                "success": False
            }
    
    def get_capabilities(self):
        """Return tool capability description"""
        return {
            "name": "chartmoe",
            "description": "ChartMoE: A specialized tool for chart understanding and data extraction",
            "capabilities": ["to_table", "describe", "extract_data", "summarize", "analyze", "compare", "trend", "title"],
            "tasks": {
                "to_table": "Convert chart to table format",
                "describe": "Describe chart in detail",
                "extract_data": "Extract numerical data",
                "summarize": "Summarize chart content",
                "analyze": "Deep analysis with insights",
                "compare": "Compare data series",
                "trend": "Identify trends",
                "title": "Extract chart title"
            },
            "supports_custom_prompt": True,
            "model_info": {
                "model_path": self.config.get('model_path', '/workspace/mathvista/model'),
                "device": self.device,
                "max_new_tokens": self.max_new_tokens,
                "image_size": self.image_size
            }
        }

# Test code
if __name__ == "__main__":
    print("\n=== Testing ChartMoE VLMGym Tool (Fixed 490x490) ===\n")
    
    # First check model path
    model_path = "/workspace/mathvista/model"
    if not os.path.exists(model_path):
        print(f"Warning: Default model path does not exist: {model_path}")
        found_path = find_chartmoe_model()
        if found_path:
            model_path = found_path
            print(f"Using discovered model path: {model_path}")
        else:
            print("Cannot find ChartMoE model")
            exit(1)
    else:
        print(f"Found model path: {model_path}")
    
    # Test config
    test_config = {
        "device": "cuda",
        "model_path": model_path,
        "trust_remote_code": True,
        "max_new_tokens": 512
    }
    
    # Create tool
    tool = FixedChartMoETool(test_config)
    
    # Find test image
    test_image = "/workspace/mathvista/data/chartqa/train/png/00006834003065.png"
    
    if not os.path.exists(test_image):
        print("Looking for other test images...")
        import glob
        png_files = glob.glob("/workspace/mathvista/data/chartqa/train/png/*.png")
        if png_files:
            test_image = png_files[0]
            print(f"Using: {test_image}")
        else:
            # Create test image
            print("Creating test image...")
            import numpy as np
            test_img = Image.fromarray(np.random.randint(0, 255, (490, 490, 3), dtype=np.uint8))
            test_img.save("/tmp/test_chart.png")
            test_image = "/tmp/test_chart.png"
    
    if os.path.exists(test_image):
        print(f"\nTest image: {test_image}")
        
        # Load image
        image = Image.open(test_image)
        print(f"Image mode: {image.mode}, size: {image.size}")
        
        tool.reset(image)
        
        # Test different tasks
        test_tasks = [
            ("describe", "Describe chart"),
            ("to_table", "Convert to table"),
            ("title", "Extract title"),
            ("summarize", "Summarize")
        ]
        
        for task, desc in test_tasks:
            print(f"\nTesting: {desc}")
            result = tool.execute(json.dumps({"task": task}))
            
            if result.get('success'):
                print("Success!")
                output = result['processed_output']
                # Limit output length
                if len(output) > 200:
                    print(f"Result: {output[:200]}...")
                else:
                    print(f"Result: {output}")
            else:
                print(f"Error: {result.get('error')}")
        
        # Test custom prompt
        print("\nTesting: Custom prompt")
        result = tool.execute(json.dumps({"prompt": "What is the highest value shown in this chart?"}))
        
        if result.get('success'):
            print("Success!")
            print(f"Result: {result['processed_output']}")
        else:
            print(f"Error: {result.get('error')}")
        
    else:
        print("Cannot find test image")
    
    # Display tool capabilities
    print("\n=== Tool Capabilities ===")
    capabilities = tool.get_capabilities()
    print(f"Name: {capabilities['name']}")
    print(f"Description: {capabilities['description']}")
    print(f"Supported tasks: {list(capabilities['tasks'].keys())}")
    print(f"Image size: {capabilities['model_info']['image_size']}x{capabilities['model_info']['image_size']}")
