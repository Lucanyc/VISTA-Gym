# chartmoe_vlmgym_tool_fixed.py
"""
ChartMoE VLMGym 工具 - 修复版（使用正确的 490x490 尺寸）
"""
from PIL import Image
import json
import os
import logging
from pathlib import Path
import torch
from torchvision import transforms

# 设置环境变量
os.environ["HF_TORCH_LOAD_DISABLE_SAFE_CHECK"] = "1"
os.environ["TRUST_REMOTE_CODE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# 应用所有修复
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

# 导入必要的模块
try:
    from vlm_gym.environments.tools.chart.chartmoe import ChartMoETool
except ImportError:
    # 如果没有安装 vlm_gym，创建一个基类
    class ChartMoETool:
        pass

from transformers import AutoModel, AutoTokenizer

# 全局变量存储模型（单例模式）
_model_instances = {}
_tokenizer_instances = {}

def find_chartmoe_model():
    """自动查找 ChartMoE 模型路径"""
    possible_paths = [
        "/workspace/mathvista/model",  # 实际模型位置
        "/workspace/model",
        "/workspace/chartmoe",
        "/workspace/models/chartmoe",
        "/models/chartmoe",
    ]
    
    for path in possible_paths:
        model_path = Path(path)
        if model_path.exists() and model_path.is_dir():
            # 检查是否包含必要的模型文件
            if (model_path / "config.json").exists():
                print(f"✓ 找到 ChartMoE 模型: {path}")
                return str(model_path.absolute())
    
    return None

def get_model_and_tokenizer(config):
    """获取或创建模型实例（单例）"""
    global _model_instances, _tokenizer_instances
    
    # 获取模型标识符
    model_identifier = config.get('model_path') or config.get('model_name')
    
    # 如果没有指定路径，尝试自动查找
    if not model_identifier:
        found_path = find_chartmoe_model()
        if found_path:
            model_identifier = found_path
            print(f"使用自动发现的模型路径: {model_identifier}")
        else:
            model_identifier = "/workspace/mathvista/model"
            print(f"使用默认模型路径: {model_identifier}")
    
    # 检查是否是本地路径
    model_path = Path(model_identifier)
    is_local = model_path.exists() and model_path.is_dir()
    
    # 使用模型路径作为键
    cache_key = str(model_path.absolute()) if is_local else model_identifier
    
    if cache_key not in _model_instances:
        print(f"首次加载 ChartMoE 模型: {model_identifier}")
        
        # 确定设备
        device = config.get('device', 'cuda')
        if device == 'cpu':
            device_map = 'cpu'
        else:
            device_map = device if ':' in device else f"{device}:0"
        
        # 准备加载参数
        load_kwargs = {
            "trust_remote_code": config.get('trust_remote_code', True),
            "torch_dtype": torch.float16 if device != 'cpu' else torch.float32,
            "device_map": device_map,
        }
        
        # 加载本地模型
        if is_local:
            model_path_str = str(model_path.absolute())
            print(f"检测到本地模型路径: {model_path_str}")
            
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
            raise ValueError(f"ChartMoE 模型未找到在路径: {model_identifier}")
        
        print("✓ 模型加载完成")
    
    return _model_instances[cache_key], _tokenizer_instances[cache_key]

class FixedChartMoETool(ChartMoETool):
    def __init__(self, config):
        # 初始化基本属性
        self.config = config
        self.device = config.get('device', 'cuda')
        self.current_image = None
        self.model = None
        self.tokenizer = None
        
        # 设置日志
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 延迟加载模型
        self._model_loaded = False
        
        # 从配置中获取参数
        self.max_new_tokens = config.get('max_new_tokens', 800)
        
        # ChartMoE 使用 490x490 的图像尺寸！
        self.image_size = 490
    
    def _load_model(self):
        """延迟加载模型"""
        if not self._model_loaded:
            try:
                self.model, self.tokenizer = get_model_and_tokenizer(self.config)
                self._model_loaded = True
                print(f"✓ 模型加载成功，使用图像尺寸: {self.image_size}x{self.image_size}")
            except Exception as e:
                self.logger.error(f"模型加载失败: {str(e)}")
                raise
    
    def reset(self, image):
        """重置工具with新图片"""
        # 确保图片是 PIL Image
        if isinstance(image, str):
            image = Image.open(image)
        
        # 关键：确保转换为 RGB（去除 alpha 通道）
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        self.current_image = image
        self.logger.info(f"Reset with image size: {image.size}, mode: {image.mode}")
    
    def execute(self, params_str):
        """执行图表分析任务"""
        try:
            # 确保模型已加载
            self._load_model()
            
            # 解析参数
            params = json.loads(params_str) if isinstance(params_str, str) else params_str
            
            task = params.get('task', '')
            prompt = params.get('prompt', '')
            
            # 任务映射
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
            
            # 处理图像 - 使用 490x490！
            rgb_image = self.current_image.convert('RGB') if self.current_image.mode != 'RGB' else self.current_image
            
            # 创建预处理管道 - 使用 490x490
            preprocess = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073], 
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])
            
            # 应用预处理
            image_tensor = preprocess(rgb_image).unsqueeze(0).to(self.model.device)
            
            # 根据设备类型调整数据类型
            if self.device != 'cpu':
                image_tensor = image_tensor.half()
            
            self.logger.info(f"Processed image tensor shape: {image_tensor.shape}, device: {image_tensor.device}")
            
            # 调用模型
            try:
                response = self.model.chat(
                    tokenizer=self.tokenizer,
                    query=formatted_query,
                    image=image_tensor,
                    history=[],
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False
                )
                
                # 处理响应（chat 返回元组）
                if isinstance(response, tuple):
                    response = response[0]
                
                return {
                    "processed_output": response,
                    "task_type": task,
                    "success": True
                }
                
            except Exception as e:
                self.logger.error(f"Model inference failed: {str(e)}")
                # 如果失败，尝试使用更小的 max_new_tokens
                if "out of memory" in str(e).lower():
                    self.logger.info("尝试使用更少的 tokens...")
                    response = self.model.chat(
                        tokenizer=self.tokenizer,
                        query=formatted_query,
                        image=image_tensor,
                        history=[],
                        max_new_tokens=200,  # 减少 tokens
                        do_sample=False
                    )
                    
                    if isinstance(response, tuple):
                        response = response[0]
                    
                    return {
                        "processed_output": response,
                        "task_type": task,
                        "success": True,
                        "note": "使用了减少的 max_new_tokens"
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
        """返回工具的能力描述"""
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

# 测试代码
if __name__ == "__main__":
    print("\n=== 测试 ChartMoE VLMGym 工具（修复版 490x490）===\n")
    
    # 首先检查模型路径
    model_path = "/workspace/mathvista/model"
    if not os.path.exists(model_path):
        print(f"⚠️  警告: 默认模型路径不存在: {model_path}")
        found_path = find_chartmoe_model()
        if found_path:
            model_path = found_path
            print(f"✓ 使用找到的模型路径: {model_path}")
        else:
            print("❌ 无法找到 ChartMoE 模型")
            exit(1)
    else:
        print(f"✓ 找到模型路径: {model_path}")
    
    # 测试配置
    test_config = {
        "device": "cuda",
        "model_path": model_path,
        "trust_remote_code": True,
        "max_new_tokens": 512
    }
    
    # 创建工具
    tool = FixedChartMoETool(test_config)
    
    # 查找测试图片
    test_image = "/workspace/mathvista/data/chartqa/train/png/00006834003065.png"
    
    if not os.path.exists(test_image):
        print("查找其他测试图片...")
        import glob
        png_files = glob.glob("/workspace/mathvista/data/chartqa/train/png/*.png")
        if png_files:
            test_image = png_files[0]
            print(f"使用: {test_image}")
        else:
            # 创建测试图片
            print("创建测试图片...")
            import numpy as np
            test_img = Image.fromarray(np.random.randint(0, 255, (490, 490, 3), dtype=np.uint8))
            test_img.save("/tmp/test_chart.png")
            test_image = "/tmp/test_chart.png"
    
    if os.path.exists(test_image):
        print(f"\n测试图片: {test_image}")
        
        # 加载图片
        image = Image.open(test_image)
        print(f"图片模式: {image.mode}, 尺寸: {image.size}")
        
        tool.reset(image)
        
        # 测试不同的任务
        test_tasks = [
            ("describe", "描述图表"),
            ("to_table", "转换为表格"),
            ("title", "提取标题"),
            ("summarize", "总结")
        ]
        
        for task, desc in test_tasks:
            print(f"\n测试: {desc}")
            result = tool.execute(json.dumps({"task": task}))
            
            if result.get('success'):
                print("✅ 成功!")
                output = result['processed_output']
                # 限制输出长度
                if len(output) > 200:
                    print(f"结果: {output[:200]}...")
                else:
                    print(f"结果: {output}")
            else:
                print(f"❌ 错误: {result.get('error')}")
        
        # 测试自定义提示
        print("\n测试: 自定义提示")
        result = tool.execute(json.dumps({"prompt": "What is the highest value shown in this chart?"}))
        
        if result.get('success'):
            print("✅ 成功!")
            print(f"结果: {result['processed_output']}")
        else:
            print(f"❌ 错误: {result.get('error')}")
        
    else:
        print("❌ 找不到测试图片")
    
    # 显示工具能力
    print("\n=== 工具能力信息 ===")
    capabilities = tool.get_capabilities()
    print(f"名称: {capabilities['name']}")
    print(f"描述: {capabilities['description']}")
    print(f"支持的任务: {list(capabilities['tasks'].keys())}")
    print(f"图像尺寸: {capabilities['model_info']['image_size']}x{capabilities['model_info']['image_size']}")
