"""
4KAgent Super-Resolution Tool for VERL Framework
Image super-resolution using 4KAgent's various SR models
"""

import sys
import json
import re
import os
import logging
import shutil
import tempfile
from typing import Tuple, Dict, Any, Optional, List
from pathlib import Path
import numpy as np
from datetime import datetime

import ray
import torch
from PIL import Image

# 设置环境变量
os.environ["HF_TORCH_LOAD_DISABLE_SAFE_CHECK"] = "1"
os.environ["TRUST_REMOTE_CODE"] = "1"

# 避免导入整个verl_tool，只导入需要的部分
from .base import BaseTool, register_tool

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. 定义 Ray Actor - 4KAgent模型的持有者
# ==============================================================================
@ray.remote(num_gpus=1 if torch.cuda.is_available() else 0)
class Agent4KResolutionActor:
    """Ray Actor for 4KAgent Super-Resolution models"""
    
    def __init__(self, fourkagent_dir: str = "/projects/slmreasoning/meng/ACL26-VLM/medical/VlmGym/tools/4KAgent",
                 output_dir: str = "/projects/slmreasoning/meng/ACL26-VLM/medical/VlmGym/classified_by_tools/tool_excuted"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Initializing Agent4KResolutionActor on PID: {os.getpid()}, Device: {self.device}")
        
        # 设置4KAgent路径
        self.fourkagent_dir = Path(fourkagent_dir)
        
        # 设置输出目录
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output directory: {self.output_dir}")
        
        # 保存原始工作目录
        self.original_cwd = os.getcwd()
        
        # 加载executor
        self.executor = self._load_executor()
        
        # 支持的工具列表
        self.supported_tools = {
            "hat_psnr": "HAT model optimized for PSNR (4x upscale)",
            "hat_gan": "HAT model optimized for visual quality (4x upscale)", 
            "hat_psnr_2x": "HAT model optimized for PSNR (2x upscale)",
            "swinfir": "SwinFIR model (4x upscale)",
            "swinfir_2x": "SwinFIR model (2x upscale)",
            "diffbir": "DiffBIR diffusion-based SR (4x upscale)",
            "diffbir_2x": "DiffBIR diffusion-based SR (2x upscale)",
            "osediff": "OSEDiff one-step diffusion SR (4x upscale)",
            "osediff_2x": "OSEDiff one-step diffusion SR (2x upscale)",
            "pisasr": "PiSA-SR model (4x upscale)",
            "pisasr_2x": "PiSA-SR model (2x upscale)",
            "hma": "HMANet model (4x upscale)",
            "hma_2x": "HMANet model (2x upscale)",
        }
        
        if self.executor:
            self.logger.info("✓ 4KAgent SR Actor initialized successfully.")
            self.logger.info(f"Available tools: {list(self.supported_tools.keys())}")
        else:
            self.logger.error("❌ 4KAgent SR Actor failed to initialize.")
    
    def _load_executor(self):
        """Load 4KAgent executor"""
        self.logger.info(f"Loading 4KAgent executor from: {self.fourkagent_dir}")
        
        try:
            # 切换到4KAgent目录
            os.chdir(self.fourkagent_dir)
            
            # 添加4KAgent到sys.path
            if str(self.fourkagent_dir) not in sys.path:
                sys.path.insert(0, str(self.fourkagent_dir))
            
            # 导入executor
            from executor import executor
            
            self.logger.info("✓ 4KAgent executor loaded successfully")
            self.logger.info(f"Available subtasks: {executor.subtasks}")
            
            return executor
            
        except Exception as e:
            self.logger.error(f"Failed to load 4KAgent executor: {e}")
            import traceback
            traceback.print_exc()
            # 恢复原始目录
            os.chdir(self.original_cwd)
            return None
    
    def inference(self, image_data: np.ndarray, params: Dict[str, Any], 
                  trajectory_id: str = None, original_image_path: str = None) -> Dict[str, Any]:
        """Execute 4KAgent super-resolution inference"""
        # 添加图片哈希用于调试
        image_hash = hash(image_data.tobytes()[:1000])
        self.logger.info(f"[Agent4K] Processing image hash: {image_hash}, shape={image_data.shape}, params={params}")
        
        if not self.executor:
            return {"error": "4KAgent executor is not available in the Actor."}
        
        try:
            # 确保在4KAgent目录中执行
            os.chdir(self.fourkagent_dir)
            
            # 获取参数
            tool_name = params.get('tool', 'hat_psnr')
            scale = params.get('scale', 4)  # 2, 4, or 16
            task = params.get('task', 'super-resolution')
            save_output = params.get('save_output', True)  # 默认保存输出
            
            # 验证工具名
            if tool_name not in self.supported_tools:
                return {
                    "error": f"Unsupported tool: {tool_name}. Available tools: {list(self.supported_tools.keys())}",
                    "success": False
                }
            
            # 根据scale选择正确的subtask
            if scale == 2 and not tool_name.endswith('_2x'):
                # 尝试使用2x版本
                if f"{tool_name}_2x" in self.supported_tools:
                    tool_name = f"{tool_name}_2x"
                    self.logger.info(f"[Agent4K] Auto-switched to 2x version: {tool_name}")
            elif scale == 16:
                subtask = 'super-resolution_16x'
            elif scale == 2:
                subtask = 'super-resolution_2x'
            else:
                subtask = 'super-resolution'
            
            self.logger.info(f"[Agent4K] Tool: {tool_name}, Subtask: {subtask}, Scale: {scale}x")
            
            # 创建临时目录
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                input_dir = temp_path / 'input'
                output_dir = temp_path / 'output'
                input_dir.mkdir()
                output_dir.mkdir()
                
                # 保存输入图像
                input_image_path = input_dir / 'input.png'
                Image.fromarray(image_data).save(input_image_path)
                
                # 执行超分辨率
                self.logger.info(f"[Agent4K] Invoking tool: {tool_name}")
                self.logger.info(f"[Agent4K] Current directory: {os.getcwd()}")
                start_time = os.times()
                
                # 调用executor - 确保在正确的目录
                self.executor.invoke_a_tool(
                    subtask_name=subtask,
                    tool_name=tool_name,
                    input_dir=input_dir,
                    output_dir=output_dir
                )
                
                end_time = os.times()
                processing_time = end_time.elapsed - start_time.elapsed
                
                # 读取输出图像
                output_files = list(output_dir.glob('*.png')) + list(output_dir.glob('*.jpg'))
                if not output_files:
                    return {
                        "error": "No output image generated",
                        "success": False
                    }
                
                output_image = Image.open(output_files[0])
                output_array = np.array(output_image)
                
                # 保存输出图像到指定目录
                saved_path = None
                if save_output:
                    # 生成输出文件名
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # 如果有原始图片路径，使用原始文件名作为基础
                    if original_image_path:
                        original_name = Path(original_image_path).stem
                        output_filename = f"{original_name}_{tool_name}_{scale}x_{timestamp}.png"
                    else:
                        output_filename = f"{trajectory_id or 'output'}_{tool_name}_{scale}x_{timestamp}.png"
                    
                    saved_path = self.output_dir / output_filename
                    output_image.save(saved_path)
                    self.logger.info(f"[Agent4K] Output image saved to: {saved_path}")
                
                # 构建结果
                result = {
                    "tool_used": tool_name,
                    "subtask": subtask,
                    "scale": scale,
                    "input_shape": list(image_data.shape),
                    "output_shape": list(output_array.shape),
                    "processing_time": float(processing_time),
                    "success": True,
                    "saved_path": str(saved_path) if saved_path else None,
                    "output_image": output_array.tolist() if output_array.size < 100000 else None,  # 只对小图返回数据
                    "output_summary": {
                        "width": output_array.shape[1],
                        "height": output_array.shape[0],
                        "channels": output_array.shape[2] if len(output_array.shape) > 2 else 1,
                        "dtype": str(output_array.dtype),
                        "size_increase": f"{output_array.shape[0]/image_data.shape[0]:.1f}x"
                    }
                }
                
                self.logger.info(f"[Agent4K] Super-resolution completed: "
                               f"{image_data.shape} -> {output_array.shape}, "
                               f"time: {processing_time:.2f}s")
                
                return result
            
        except Exception as e:
            error_msg = str(e)
            import traceback
            error_trace = traceback.format_exc()
            self.logger.error(f"Agent4K inference failed: {error_msg}")
            self.logger.error(f"Traceback: {error_trace}")
            return {
                "error": error_msg,
                "error_details": error_trace,
                "success": False
            }
        finally:
            # 总是尝试恢复到原始目录
            try:
                os.chdir(self.original_cwd)
            except:
                pass
    
    def health_check(self) -> str:
        """健康检查"""
        if self.executor is not None:
            return f"healthy: 4KAgent executor loaded with {len(self.supported_tools)} tools"
        else:
            return "unhealthy: executor not loaded"

# ==============================================================================
# 2. Agent4KResolutionTool - 工具接口
# ==============================================================================
@register_tool
class Agent4KResolutionTool(BaseTool):
    tool_type = "agent4k_resolution"
    
    def __init__(self, num_workers=1, fourkagent_dir=None, output_dir=None, **kwargs):
        super().__init__(num_workers)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 获取4KAgent目录
        self.fourkagent_dir = fourkagent_dir or os.environ.get(
            'FOURKAGENT_DIR',
            "/projects/slmreasoning/meng/ACL26-VLM/medical/VlmGym/tools/4KAgent"
        )
        
        # 获取输出目录
        self.output_dir = output_dir or os.environ.get(
            'AGENT4K_OUTPUT_DIR',
            "/projects/slmreasoning/meng/ACL26-VLM/medical/VlmGym/classified_by_tools/tool_excuted"
        )
        
        self.logger.info(f"Using 4KAgent directory: {self.fourkagent_dir}")
        self.logger.info(f"Using output directory: {self.output_dir}")
        
        # 验证路径
        if not os.path.exists(self.fourkagent_dir):
            self.logger.warning(f"4KAgent directory does not exist: {self.fourkagent_dir}")
        
        # 延迟初始化Ray和Actor - 不在__init__中初始化，避免阻塞健康检查
        # Ray和Actor将在第一次使用时初始化
        self.actor = None
        self._actor_ready = False
        self._ray_initialized = False
        self.logger.info("Agent4K Resolution Tool initialized (Ray and Actor will be initialized on first use)")
    
    def _initialize_ray_and_actor(self):
        """初始化Ray和Actor（延迟初始化）"""
        if self._ray_initialized and self.actor is not None:
            return True
        
        # 初始化Ray（如果还没有初始化）
        if not self._ray_initialized:
            self.logger.info("Initializing Ray (lazy initialization)...")
            try:
                if not ray.is_initialized():
                    # 首先尝试连接到现有的Ray集群（如果主进程已初始化）
                    try:
                        ray.init(
                            address="auto",
                            ignore_reinit_error=True,
                            log_to_driver=False,
                            include_dashboard=False,
                        )
                        self.logger.info("✓ Ray connected to existing cluster")
                        self._ray_initialized = True
                    except Exception as connect_error:
                        # 如果连接失败，启动本地Ray实例
                        # 注意：这里可能会因为版本不匹配而失败，直接启动本地实例
                        self.logger.info(f"Could not connect to existing Ray cluster, starting local instance...")
                        ray.init(
                            ignore_reinit_error=True,
                            log_to_driver=False,
                            include_dashboard=False,
                        )
                        self.logger.info("✓ Ray initialized with local instance")
                        self._ray_initialized = True
                else:
                    self.logger.info("✓ Ray already initialized, reusing existing cluster")
                    self._ray_initialized = True
            except Exception as e:
                self.logger.error(f"Failed to initialize Ray: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                return False
        
        # 创建Actor（如果还没有创建）
        if self.actor is None:
            self.logger.info("Creating Agent4K Resolution Actor...")
            try:
                num_gpus = 1 if torch.cuda.is_available() else 0
                self.logger.info(f"Creating actor with num_gpus={num_gpus}")
                
                actor_ref = Agent4KResolutionActor.options(
                    name="Agent4KResolutionActor",
                    get_if_exists=True,
                    num_gpus=num_gpus,
                    max_concurrency=2  # SR tasks can be memory intensive
                ).remote(fourkagent_dir=self.fourkagent_dir, output_dir=self.output_dir)
                
                # 验证actor引用是否有效
                if actor_ref is None:
                    raise ValueError("Actor reference is None after creation")
                
                self.actor = actor_ref
                self.logger.info("✓ Agent4K Resolution Actor created")
                self._actor_ready = False
                    
            except Exception as e:
                self.logger.error(f"Failed to create Agent4K Resolution Actor: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                self.actor = None
                return False
        
        return True
    
    def _ensure_actor_ready(self):
        """确保Actor已准备好（延迟健康检查）"""
        # 首先确保Ray和Actor已初始化
        if not self._initialize_ray_and_actor():
            return False
        
        if not hasattr(self, '_actor_ready') or not self._actor_ready:
            try:
                self.logger.info("Performing deferred health check on Agent4K Resolution Actor...")
                health_status = ray.get(self.actor.health_check.remote(), timeout=60)
                
                if "healthy" in health_status:
                    self.logger.info(f"✓ Agent4K Resolution Actor is {health_status}")
                    self._actor_ready = True
                    return True
                else:
                    self.logger.error(f"Agent4K Resolution Actor health check failed: {health_status}")
                    self._actor_ready = False
                    return False
            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
                self._actor_ready = False
                return False
        
        return True
    
    def get_usage_inst(self):
        """返回工具使用说明"""
        return ('Agent4K-Resolution: Image super-resolution with multiple models. '
                'Use <tool_call>{"tool": "agent4k_resolution", "model": "hat_psnr|hat_gan|swinfir|diffbir|osediff|pisasr|hma", '
                '"scale": 2|4|16, "task": "super-resolution", "save_output": true}</tool_call>')
    
    def parse_action(self, action: str) -> Tuple[Dict[str, Any], bool]:
        """解析工具调用参数"""
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.findall(tool_call_pattern, action, re.DOTALL)
        
        if not matches:
            return {}, False
        
        try:
            params = json.loads(matches[0].strip())
            
            # 验证工具名
            if params.get('tool') != 'agent4k_resolution':
                return {}, False
            
            # 支持嵌套格式
            if 'parameters' in params and isinstance(params['parameters'], dict):
                inner_params = params['parameters']
            else:
                inner_params = params
            
            # 提取所有可能的参数
            parsed = {
                'tool': inner_params.get('model', 'hat_psnr'),  # 具体的SR模型
                'scale': inner_params.get('scale', 4),
                'task': inner_params.get('task', 'super-resolution'),
                'save_output': inner_params.get('save_output', True),  # 默认保存
            }
            
            self.logger.debug(f"Parsed parameters: tool={parsed['tool']}, "
                            f"scale={parsed['scale']}x, save={parsed['save_output']}")
            
            return parsed, True
            
        except Exception as e:
            self.logger.error(f"Error parsing action: {e}")
            return {}, False
    
    def conduct_action(self, trajectory_id: str, action: str, extra_field: Dict[str, Any]) -> Tuple[str, bool, bool]:
        """执行Agent4K Resolution工具调用"""
        parsed_params, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)
        
        if not is_valid:
            observation = "Error: Invalid action format. Expected <tool_call>{...}</tool_call>"
            self.update_env(trajectory_id, env, action, False, extra_field, observation)
            self.save_env(trajectory_id, env)
            return f"\n```output\n{observation}\n```\n", False, False
        
        try:
            # 调试日志
            self.logger.info(f"[Agent4K-Resolution] Processing request - trajectory_id: {trajectory_id}")
            self.logger.debug(f"[Agent4K-Resolution] Received extra_field keys: {list(extra_field.keys()) if extra_field else 'None'}")
            
            # 获取图像和原始路径
            image, original_path = self._get_image_with_path(trajectory_id, extra_field)
            
            if image is None:
                observation = "Error: No image provided for super-resolution"
                self.logger.error(f"[Agent4K-Resolution] {observation}")
            elif not self._ensure_actor_ready():
                observation = "Error: Agent4K Resolution Actor is not available or not ready. Please try again later."
                self.logger.error(observation)
            else:
                # 使用parsed_params中的参数
                self.logger.info(f"[Agent4K-Resolution] Processing with tool: {parsed_params['tool']}, "
                               f"scale: {parsed_params['scale']}x")
                
                image_array = np.array(image.convert('RGB'))
                
                # 传递完整的parsed_params到Actor，包括trajectory_id和原始图片路径
                result_ref = self.actor.inference.remote(
                    image_array, 
                    parsed_params,
                    trajectory_id=trajectory_id,
                    original_image_path=original_path
                )
                try:
                    result = ray.get(result_ref, timeout=620)  # SR can take longer
                    
                    if result.get('success', False):
                        observation = self._format_output(result)
                    else:
                        observation = f"Error: {result.get('error', 'Unknown error')}"
                    
                    self.logger.info(f"[Agent4K-Resolution] Successfully processed image for trajectory {trajectory_id}")
                    
                except ray.exceptions.GetTimeoutError:
                    observation = "Error: Agent4K Resolution inference timed out (600s)"
                    self.logger.error(observation)
            
            self.update_env(trajectory_id, env, parsed_params, is_valid, extra_field, observation)
            self.save_env(trajectory_id, env)
            return f"\n```output\n{observation}\n```\n", False, True
            
        except Exception as e:
            self.logger.error(f"Agent4K Resolution tool conduct_action failed: {e}", exc_info=True)
            observation = f"Error: {str(e)}"
            return f"\n```output\n{observation}\n```\n", False, True
    
    def _get_image_with_path(self, trajectory_id: str, extra_field: Dict[str, Any]) -> Tuple[Optional[Image.Image], Optional[str]]:
        """从extra_field获取图像和原始路径"""
        
        # 只从extra_field['images']获取图片
        if extra_field and 'images' in extra_field:
            images_list = extra_field['images']
            self.logger.info(f"[Agent4K-Resolution] Found 'images' in extra_field: {images_list}")
            
            if images_list:
                # 获取第一个图片路径
                img_path = images_list[0] if isinstance(images_list, list) else images_list
                
                if img_path and isinstance(img_path, str):
                    if os.path.exists(img_path):
                        self.logger.info(f"[Agent4K-Resolution] ✅ Loading image from: {img_path}")
                        loaded_image = Image.open(img_path).convert('RGB')
                        self.logger.info(f"[Agent4K-Resolution] Image size: {loaded_image.size}")
                        return loaded_image, img_path
                    else:
                        self.logger.error(f"[Agent4K-Resolution] ❌ Image path does not exist: {img_path}")
                else:
                    self.logger.error(f"[Agent4K-Resolution] ❌ Invalid image path format: {type(img_path)} - {img_path}")
            else:
                self.logger.error(f"[Agent4K-Resolution] ❌ images list is empty")
        
        # 没有找到图片
        self.logger.error(f"[Agent4K-Resolution] ❌ NO IMAGE PROVIDED!")
        self.logger.error(f"[Agent4K-Resolution] Available extra_field keys: {list(extra_field.keys()) if extra_field else 'None'}")
        return None, None
    
    def _format_output(self, result: Dict[str, Any]) -> str:
        """格式化输出结果"""
        if 'error' in result:
            return f"Error: {result['error']}"
        
        output_lines = []
        
        # 总体结果
        output_lines.append(f"Agent4K Super-Resolution Results:")
        output_lines.append(f"  Tool used: {result.get('tool_used', 'N/A')}")
        output_lines.append(f"  Scale factor: {result.get('scale', 'N/A')}x")
        output_lines.append(f"  Input shape: {result.get('input_shape', 'N/A')}")
        output_lines.append(f"  Output shape: {result.get('output_shape', 'N/A')}")
        output_lines.append(f"  Processing time: {result.get('processing_time', 0):.2f}s")
        
        # 输出摘要
        summary = result.get('output_summary', {})
        if summary:
            output_lines.append(f"\nOutput Details:")
            output_lines.append(f"  Resolution: {summary.get('width')}x{summary.get('height')}")
            output_lines.append(f"  Size increase: {summary.get('size_increase', 'N/A')}")
        
        # 保存的文件路径
        if result.get('saved_path'):
            output_lines.append(f"\n✅ Output image saved to:")
            output_lines.append(f"  {result['saved_path']}")
        
        return '\n'.join(output_lines)
    
    def update_env(self, trajectory_id: str, env: Dict[str, Any], action: Any, 
                   is_valid: bool, extra_field: Dict[str, Any], observation: str, **kwargs):
        """更新环境"""
        # 只调用父类的update_env
        super().update_env(trajectory_id, env, action, is_valid, extra_field, observation, **kwargs)