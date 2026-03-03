# vlm_gym/environments/tools/medsam2.py
import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional, Union
import cv2
import subprocess
import textwrap

# 添加MedSAM2路径
MEDSAM2_PATH = '/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/MedSAM2'
sys.path.append(MEDSAM2_PATH)

from vlm_gym.environments.tools.base import ToolBase


class MedSAM2Tool(ToolBase):
    name = "medsam2"
    """
    MedSAM2 医疗图像分割工具
    支持3D CT/MRI图像分割和医疗视频分割
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.name = "medsam2"
        self.description = "医疗图像和视频分割工具，支持3D CT/MRI分割"
        self.capabilities = ["3D医疗图像分割", "医疗视频分割", "病灶检测", "器官分割"]
        
        # 调用父类初始化
        super().__init__(name=self.name)
        
        # 配置
        self.config = config or {}
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # MedSAM2路径配置
        self.medsam2_path = MEDSAM2_PATH
        # 使用相对路径（相对于MedSAM2目录）
        self.checkpoint_filename = "checkpoints/MedSAM2_latest.pt"
        # 尝试不同的配置文件
        self.model_cfg_name = "sam2.1_hiera_t512.yaml"
        
        # 模型延迟加载
        self.predictor = None
        self.current_image = None
        self.current_image_np = None
        self.current_frames = None  # 用于3D图像的帧序列
        self.inference_state = None
        
        print(f"[MedSAM2] Initialized with device: {self.device}")
        print(f"[MedSAM2] MedSAM2 path: {self.medsam2_path}")
    
    def _load_model(self):
        """延迟加载模型（最终工作方案）"""
        if self.predictor is None:
            print("[MedSAM2] Loading model...")
            
            original_cwd = os.getcwd()
            
            try:
                # 使用之前测试成功的方法
                print("[MedSAM2] Using verified loading method...")
                
                # 创建一个简单的独立加载脚本，使用textwrap.dedent去除缩进
                load_script = textwrap.dedent(f"""
                import os
                import sys
                import torch
                
                # 确保在正确的目录
                os.chdir('{self.medsam2_path}')
                sys.path.insert(0, os.getcwd())
                
                from sam2.build_sam import build_sam2_video_predictor_npz
                
                # 使用configs目录下的配置文件（不带前缀路径）
                predictor = build_sam2_video_predictor_npz(
                    "configs/sam2.1_hiera_t512.yaml",
                    "checkpoints/MedSAM2_latest.pt",
                    device="{self.device}"
                )
                
                # 保存predictor
                torch.save(predictor, "temp_predictor_output.pt")
                print("SUCCESS")
                """).strip()
                
                # 写入脚本
                script_path = os.path.join(self.medsam2_path, "load_medsam2_standalone.py")
                with open(script_path, 'w') as f:
                    f.write(load_script)
                
                # 执行脚本
                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    cwd=self.medsam2_path
                )
                
                if "SUCCESS" in result.stdout:
                    # 加载保存的predictor
                    predictor_path = os.path.join(self.medsam2_path, "temp_predictor_output.pt")
                    # 设置weights_only=False以允许加载包含自定义类的文件
                    self.predictor = torch.load(predictor_path, map_location=self.device, weights_only=False)
                    
                    # 清理临时文件
                    os.remove(script_path)
                    os.remove(predictor_path)
                    
                    print("[MedSAM2] Model loaded successfully!")
                    return
                else:
                    print(f"[MedSAM2] Loading failed:")
                    print(f"STDOUT: {result.stdout}")
                    print(f"STDERR: {result.stderr}")
                    
            except Exception as e:
                print(f"[MedSAM2] Error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                os.chdir(original_cwd)
            
            raise RuntimeError("Failed to load MedSAM2 model")
    
    def _prepare_image(self, image, target_size=None, preserve_aspect_ratio=True):
        """
        准备图像以符合MedSAM2的要求（尺寸必须是32的倍数）
        
        Args:
            image: PIL Image 或 numpy array
            target_size: 目标尺寸 (width, height)，如果为None则自动计算
            preserve_aspect_ratio: 是否保持纵横比
            
        Returns:
            tuple: (processed_array, original_size, resize_info)
        """
        # 如果是numpy数组，获取其尺寸
        if isinstance(image, np.ndarray):
            original_size = (image.shape[1], image.shape[0])  # (width, height)
            img_array = image
        else:
            # PIL Image
            original_size = image.size
            img_array = np.array(image)
        
        h, w = img_array.shape[:2]
        
        if target_size is None:
            # 自动计算最近的32倍数尺寸
            if preserve_aspect_ratio:
                # 保持纵横比
                max_dim = max(w, h)
                # 向上取整到最近的32的倍数
                target_max = ((max_dim + 31) // 32) * 32
                
                # 限制最大尺寸避免内存问题
                if target_max > 1024:
                    target_max = 1024
                elif target_max < 128:  # 太小的图像放大
                    target_max = 128
                
                # 计算缩放比例
                scale = target_max / max_dim
                target_w = ((int(w * scale) + 31) // 32) * 32
                target_h = ((int(h * scale) + 31) // 32) * 32
            else:
                # 不保持纵横比，直接调整到最近的32倍数
                target_w = ((w + 31) // 32) * 32
                target_h = ((h + 31) // 32) * 32
        else:
            target_w, target_h = target_size
            # 确保目标尺寸是32的倍数
            target_w = ((target_w + 31) // 32) * 32
            target_h = ((target_h + 31) // 32) * 32
        
        # 使用cv2进行resize
        if (h, w) != (target_h, target_w):
            resized_array = cv2.resize(img_array, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized_array = img_array
        
        resize_info = {
            'original_size': original_size,
            'target_size': (target_w, target_h),
            'scale_factor': (target_w / original_size[0], target_h / original_size[1])
        }
        
        return resized_array, original_size, resize_info
    
    def _restore_original_size(self, mask, resize_info):
        """
        将mask恢复到原始图像尺寸
        
        Args:
            mask: numpy array (H, W) 
            resize_info: resize信息字典
            
        Returns:
            numpy array: 恢复到原始尺寸的mask
        """
        original_w, original_h = resize_info['original_size']
        
        # 确保mask是2D的
        if mask.ndim > 2:
            mask = mask.squeeze()
        
        # 使用cv2进行resize，INTER_NEAREST保持二值mask的特性
        mask_restored = cv2.resize(
            mask.astype(np.uint8), 
            (original_w, original_h), 
            interpolation=cv2.INTER_NEAREST
        )
        
        return mask_restored
    
    def reset(self, image: Union[Image.Image, str, List[Image.Image], np.ndarray]):
        """
        重置工具状态，准备处理新图像
        
        Args:
            image: 可以是：
                - PIL Image（单张2D图像）
                - 图像路径
                - PIL Image列表（3D图像的切片序列）
                - numpy数组（H,W,D）表示3D体积
        """
        self._load_model()
        
        # 处理不同类型的输入
        if isinstance(image, str):
            # 如果是路径，尝试加载
            if image.endswith(('.nii', '.nii.gz')):
                # 3D医疗图像格式
                self._load_3d_volume(image)
            else:
                # 普通2D图像
                image = Image.open(image)
                self.current_image = image.convert("RGB")
                # 调整图像尺寸以符合模型要求
                processed_array, orig_size, resize_info = self._prepare_image(self.current_image)
                self.current_image_np = processed_array
                self.resize_info = resize_info  # 保存resize信息
                self.current_frames = [self.current_image_np]
                self.resize_info_list = [self.resize_info]  # 保存为列表格式
                
        elif isinstance(image, Image.Image):
            # 单张PIL图像
            self.current_image = image.convert("RGB")
            # 调整图像尺寸以符合模型要求
            processed_array, orig_size, resize_info = self._prepare_image(self.current_image)
            self.current_image_np = processed_array
            self.resize_info = resize_info  # 保存resize信息
            self.current_frames = [self.current_image_np]
            self.resize_info_list = [self.resize_info]  # 保存为列表格式
            
        elif isinstance(image, list) and all(isinstance(img, Image.Image) for img in image):
            # PIL图像列表（3D切片）
            # 处理每一帧
            self.current_frames = []
            self.resize_info_list = []
            for img in image:
                img_rgb = img.convert("RGB")
                processed_array, orig_size, resize_info = self._prepare_image(img_rgb)
                self.current_frames.append(processed_array)
                self.resize_info_list.append(resize_info)
            self.resize_info = self.resize_info_list[len(image)//2]  # 中间帧的resize信息
            self.current_image = image[len(image)//2]  # 中间切片作为代表
            self.current_image_np = self.current_frames[len(image)//2]
            
        elif isinstance(image, np.ndarray):
            # numpy数组
            if image.ndim == 3:
                # H,W,D -> 转换为帧序列
                self._process_3d_volume(image)
            elif image.ndim == 2:
                # 单张灰度图
                self.current_image_np = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                self.current_frames = [self.current_image_np]
                self.resize_info_list = [self.resize_info]  # 保存为列表格式
                self.current_image = Image.fromarray(self.current_image_np)
        
        # 初始化推理状态
        if self.predictor is not None and self.current_frames is not None:
            # 需要在MedSAM2目录下运行
            original_cwd = os.getcwd()
            os.chdir(self.medsam2_path)
            
            try:
                # 获取图像尺寸
                if len(self.current_frames) > 0:
                    video_height, video_width = self.current_frames[0].shape[:2]
                    
                    # 将numpy数组转换为PyTorch张量
                    # 从 [num_frames, H, W, C] 转换为 [num_frames, C, H, W]
                    frames_np = np.stack(self.current_frames)  # [N, H, W, C]
                    frames_np = frames_np.transpose(0, 3, 1, 2)  # [N, C, H, W]
                    frames_tensor = torch.from_numpy(frames_np).float()
                    
                    # 使用正确的参数调用init_state
                    self.inference_state = self.predictor.init_state(
                        images=frames_tensor,  # PyTorch张量 [N, C, H, W]
                        video_height=video_height,
                        video_width=video_width
                    )
                    print(f"[MedSAM2] Initialized with {len(self.current_frames)} frames, size: {video_width}x{video_height}")
            finally:
                os.chdir(original_cwd)
    
    def _load_3d_volume(self, path: str):
        """加载3D医疗图像（.nii, .nii.gz等）"""
        try:
            import SimpleITK as sitk
            # 读取3D图像
            image = sitk.ReadImage(path)
            image_np = sitk.GetArrayFromImage(image)  # (D, H, W)
            
            # 转换为帧序列
            self._process_3d_volume(image_np)
            
        except ImportError:
            raise ImportError("SimpleITK is required for loading 3D medical images")
        except Exception as e:
            raise RuntimeError(f"Failed to load 3D volume: {e}")
    
    def _process_3d_volume(self, volume: np.ndarray):
        """处理3D体积数据，转换为帧序列"""
        # 假设输入是 (D, H, W) 或 (H, W, D)
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {volume.shape}")
        
        # 标准化到0-255
        volume = ((volume - volume.min()) / (volume.max() - volume.min()) * 255).astype(np.uint8)
        
        # 转换为RGB帧序列
        frames = []
        for i in range(volume.shape[0]):
            slice_2d = volume[i]
            # 转换为RGB
            slice_rgb = cv2.cvtColor(slice_2d, cv2.COLOR_GRAY2RGB)
            frames.append(slice_rgb)
        
        self.current_frames = frames
        # 使用中间切片作为代表
        mid_idx = len(frames) // 2
        self.current_image_np = frames[mid_idx]
        self.current_image = Image.fromarray(self.current_image_np)
    
    def execute(self, action_string: str) -> Dict[str, Any]:
        """
        执行分割任务
        
        参数格式：
        {
            "task": "segment",  # 任务类型
            "prompt_type": "box",  # 提示类型：box, point, mask
            "prompts": [...],  # 提示内容
            "frame_idx": 0,  # 帧索引（用于3D/视频）
            "propagate": true  # 是否传播到其他帧
        }
        """
        if self.current_frames is None:
            return {"error": "No image loaded. Please call reset() first."}
        
        # 解析参数
        if isinstance(action_string, str):
            try:
                params = json.loads(action_string)
            except json.JSONDecodeError:
                params = {"task": "segment"}
        else:
            params = action_string
        
        task = params.get("task", "segment")
        
        if task == "segment":
            return self._segment(params)
        else:
            return {"error": f"Unknown task: {task}"}
    
    def _segment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行分割"""
        # 需要在MedSAM2目录下运行
        original_cwd = os.getcwd()
        os.chdir(self.medsam2_path)
        
        try:
            prompt_type = params.get("prompt_type", "box")
            prompts = params.get("prompts", [])
            frame_idx = params.get("frame_idx", 0)
            propagate = params.get("propagate", False)
            
            if not prompts:
                return {"error": "No prompts provided"}
            
            # 确保frame_idx有效
            if frame_idx >= len(self.current_frames):
                return {"error": f"Invalid frame index {frame_idx}, total frames: {len(self.current_frames)}"}
            
            # 准备提示
            if prompt_type == "box":
                # prompts格式: [[x1,y1,x2,y2], ...]
                points = []
                labels = []
                for box in prompts:
                    if len(box) != 4:
                        continue
                    x1, y1, x2, y2 = box
                    # 添加框的角点
                    points.extend([[x1, y1], [x2, y2]])
                    labels.extend([2, 3])  # 2=top-left, 3=bottom-right
                
                points = np.array(points)
                labels = np.array(labels)
                
            elif prompt_type == "point":
                # prompts格式: [[x,y,label], ...]
                points = []
                labels = []
                for p in prompts:
                    if len(p) >= 2:
                        points.append([p[0], p[1]])
                        labels.append(p[2] if len(p) > 2 else 1)
                
                points = np.array(points)
                labels = np.array(labels)
                
            else:
                return {"error": f"Unsupported prompt type: {prompt_type}"}
            
            # 添加提示并预测
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=1,  # 单个对象
                points=points,
                labels=labels,
            )
            
            # 获取当前帧的分割结果
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                self.inference_state,
                start_frame_idx=frame_idx,
                max_frame_num_to_track=1 if not propagate else len(self.current_frames),
            ):
                mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                video_segments[out_frame_idx] = mask
            
            # 准备返回结果
            result = {
                "success": True,
                "task": "segment",
                "num_frames": len(self.current_frames),
                "segmented_frames": list(video_segments.keys()),
                "masks": {}
            }
            
            # 转换mask为列表格式（用于JSON序列化）
            for idx, mask in video_segments.items():
                # 确保mask是2D的
                if mask.ndim > 2:
                    mask = mask.squeeze()  # 移除额外的维度
                
                # 转换为uint8类型
                mask_uint8 = mask.astype(np.uint8)
                
                # 转换为RLE或边界轮廓以减少数据量
                contours, _ = cv2.findContours(
                    mask_uint8, 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                # 恢复到原始尺寸
                if hasattr(self, "resize_info_list") and idx < len(self.resize_info_list):
                    mask = self._restore_original_size(mask, self.resize_info_list[idx])
                elif hasattr(self, "resize_info"):
                    mask = self._restore_original_size(mask, self.resize_info)
                
                result["masks"][str(idx)] = {
                    "shape": mask.shape,
                    "num_contours": len(contours),
                    "area": int(mask.sum()),
                    "mask": mask,  # 实际的mask数据
                    # 可选：包含轮廓点
                    # "contours": [c.tolist() for c in contours]
                }
            
            return result
            
        except Exception as e:
            import traceback
            return {
                "error": f"Segmentation failed: {str(e)}",
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()
            }
        finally:
            # 恢复原始工作目录
            os.chdir(original_cwd)
    
    def get_capabilities(self) -> Dict[str, Any]:
        """返回工具能力描述"""
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "parameters": {
                "task": {
                    "type": "string",
                    "default": "segment",
                    "description": "任务类型"
                },
                "prompt_type": {
                    "type": "string",
                    "enum": ["box", "point", "mask"],
                    "default": "box",
                    "description": "提示类型"
                },
                "prompts": {
                    "type": "array",
                    "required": True,
                    "description": "提示内容，格式取决于prompt_type"
                },
                "frame_idx": {
                    "type": "integer",
                    "default": 0,
                    "description": "目标帧索引（用于3D/视频）"
                },
                "propagate": {
                    "type": "boolean",
                    "default": False,
                    "description": "是否传播分割到其他帧"
                }
            },
            "output": {
                "masks": {
                    "type": "dict",
                    "description": "分割掩码，键为帧索引"
                },
                "segmented_frames": {
                    "type": "list",
                    "description": "已分割的帧索引列表"
                },
                "num_frames": {
                    "type": "integer",
                    "description": "总帧数"
                }
            }
        }