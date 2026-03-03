import sys
sys.path.append('/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista')
sys.path.append('/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/GroundingDINO')


import os
import torch
import torchvision
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Tuple, Optional

# 基于 demo 代码的实现，避免使用有问题的 inference.py
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span

from vlm_gym.environments.tools.base import ToolBase


def load_model_demo_style(model_config_path, model_checkpoint_path, cpu_only=True):
    """使用 demo 中的加载方式"""
    args = SLConfig.fromfile(model_config_path)
    args.device = "cpu" if cpu_only else "cuda"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(f"Model loading result: {load_res}")
    model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, cpu_only=True):
    """使用 demo 中的推理方式"""
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    
    device = "cpu" if cpu_only else "cuda"
    model = model.to(device)
    image = image.to(device)
    
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)
    
    # filter output
    logits_filt = logits.cpu().clone()
    boxes_filt = boxes.cpu().clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    
    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase)
        scores.append(logit.max().item())
    
    return boxes_filt, torch.tensor(scores), pred_phrases


class GroundingDINOTool(ToolBase):
    name = "grounding_dino"
    """
    Grounding DINO工具 - 使用 CPU 模式避免 C++ 扩展问题
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        # 先设置工具属性
        self.name = "grounding_dino"
        self.description = "开放词汇目标检测工具（CPU模式）"
        self.capabilities = ["目标检测", "定位", "计数", "开放词汇"]
        
        # 传递name参数给父类
        super().__init__(name=self.name)

        # 配置
        self.config = config or {}
        # 强制使用 CPU 以避免 CUDA 扩展问题
        
         # 添加调试信息
        print(f"\n[GroundingDINO.__init__] DEBUG:")
        print(f"  - config received: {config}")
        print(f"  - self.config: {self.config}")
        
        self.device = 'cpu'
        self.cpu_only = True
        print("⚠️ 使用 CPU 模式运行 GroundingDINO")
        
        # 设置默认路径
        #base_path = "/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/GroundingDINO"
        #self.model_path = self.config.get('model_path', 
        #                                 os.path.join(base_path, "groundingdino_swint_ogc.pth"))
        #self.model_config = self.config.get('model_config',
        #                                   os.path.join(base_path, "groundingdino/config/GroundingDINO_SwinT_OGC.py"))
        
        base_path = "/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/GroundingDINO"
        self.model_path = os.path.join(base_path, "groundingdino_swint_ogc.pth")
        self.model_config = os.path.join(base_path, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
    
        
        # 打印实际使用的路径
        print(f"[GroundingDINO] Using hardcoded paths:")
        print(f"  - model_path: {self.model_path}")
        print(f"  - model_config: {self.model_config}")
        
        self.default_box_threshold = self.config.get('box_threshold', 0.35)
        self.default_text_threshold = self.config.get('text_threshold', 0.25)
        self.nms_threshold = self.config.get('nms_threshold', 0.8)
        
        # 模型延迟加载
        self.model = None
        self.transform = None
        self.current_image = None
        self.current_image_np = None
        self.current_image_tensor = None
        
        
         # 验证文件是否存在
        if not os.path.exists(self.model_path):
            print(f"  - WARNING: Model file not found: {self.model_path}")
        else:
            print(f"  - ✓ Model file exists")
            
        if not os.path.exists(self.model_config):
            print(f"  - WARNING: Config file not found: {self.model_config}")
        else:
            print(f"  - ✓ Config file exists")
        
    def _load_model(self):
        """延迟加载模型"""
        if self.model is None:
            print(f"Loading Grounding DINO model from {self.model_path}")
            print(f"Using config: {self.model_config}")
            print("Note: Using CPU mode to avoid C++ extension issues")
            
            try:
                self.model = load_model_demo_style(
                    self.model_config,
                    self.model_path,
                    cpu_only=self.cpu_only
                )
                
                # 初始化图像变换
                self.transform = T.Compose([
                    T.RandomResize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
                
                print("Grounding DINO model loaded successfully (CPU mode)")
            except Exception as e:
                print(f"Error loading model: {e}")
                raise
    
    def reset(self, image: Image.Image):
        """重置工具状态，准备处理新图像"""
        self._load_model()  # 确保模型已加载
        
        # 处理输入 - 支持字符串路径或PIL Image
        if isinstance(image, str):
            image = Image.open(image)
        
        # 保存原始图像
        self.current_image = image.convert("RGB")
        
        # 准备numpy格式
        self.current_image_np = np.asarray(self.current_image)
        
        # 准备tensor格式
        self.current_image_tensor, _ = self.transform(self.current_image, None)
        
        
    def execute(self, action_string: str) -> Dict[str, Any]:
        """执行Grounding DINO检测"""
        import json
        
        if self.current_image is None:
            return {"error": "No image loaded. Please call reset() first."}
        
        # 在开始时就定义 h 和 w
        h, w = self.current_image_np.shape[:2]
        
        # 解析参数
        if isinstance(action_string, str):
            try:
                params = json.loads(action_string)
            except json.JSONDecodeError:
                # 如果不是JSON，尝试作为caption直接使用
                params = {"caption": action_string}
        else:
            params = action_string
            
        # 获取参数
        caption = params.get("caption", "")
        if not caption:
            return {"error": "Caption is required for detection"}
            
        box_threshold = params.get("box_threshold", self.default_box_threshold)
        text_threshold = params.get("text_threshold", self.default_text_threshold)
        
        try:
            # 使用 demo 风格的推理
            boxes, scores, phrases = get_grounding_output(
                model=self.model,
                image=self.current_image_tensor,
                caption=caption,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                with_logits=True,
                cpu_only=self.cpu_only
            )
            
            # 应用NMS
            if len(boxes) > 0:
                boxes, scores, phrases = self._nms(boxes, scores, phrases)
            
            # 转换边界框格式
            if len(boxes) > 0:
                # h, w 已经在上面定义了
                # 转换为像素坐标 (cxcywh -> xyxy)
                boxes = boxes * torch.Tensor([w, h, w, h])
                boxes = box_ops.box_cxcywh_to_xyxy(boxes)
                
                # 转换为列表格式
                boxes = boxes.tolist()
                boxes = [[int(round(x)) for x in box] for box in boxes]
                scores = scores.tolist()
                scores = [round(x, 3) for x in scores]
            else:
                boxes = []
                scores = []
                phrases = []
            
            result = {
                "boxes": boxes,  # [[x1,y1,x2,y2], ...] 格式，像素坐标
                "logits": scores,  # 置信度分数
                "phrases": phrases,  # 检测到的文本标签
                "size": [h, w],  # [H, W] - 现在 h, w 总是已定义
                "num_detections": len(boxes),
                "mode": "cpu"  # 标记使用 CPU 模式
            }
            
            return result
            
        except Exception as e:
            import traceback
            return {
                "error": f"Detection failed: {str(e)}",
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()
            }
    
    def _nms(self, boxes, scores, phrases):
        """应用非极大值抑制（NMS）"""
        if len(boxes) == 0:
            return boxes, scores, phrases
            
        # 确保在 CPU 上
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu()
        
        # 转换为xyxy格式进行NMS（注意这里boxes还是归一化的）
        h, w = self.current_image_np.shape[:2]
        boxes_pixel = boxes * torch.Tensor([w, h, w, h])
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_pixel)
        
        # 执行NMS
        nms_idx = torchvision.ops.nms(boxes_xyxy, scores, self.nms_threshold).numpy()
        
        # 应用NMS索引
        boxes = boxes[nms_idx]
        scores = scores[nms_idx]
        phrases = [phrases[idx] for idx in nms_idx]
        
        return boxes, scores, phrases
    
    def get_capabilities(self) -> Dict[str, Any]:
        """返回工具能力描述"""
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "mode": "cpu",
            "parameters": {
                "caption": {
                    "type": "string",
                    "required": True,
                    "description": "要检测的对象描述，支持复杂描述如'red car and blue bicycle'"
                },
                "box_threshold": {
                    "type": "float",
                    "default": self.default_box_threshold,
                    "range": [0.0, 1.0],
                    "description": "边界框置信度阈值"
                },
                "text_threshold": {
                    "type": "float",
                    "default": self.default_text_threshold,
                    "range": [0.0, 1.0],
                    "description": "文本匹配置信度阈值"
                }
            },
            "output": {
                "boxes": {
                    "type": "list",
                    "description": "检测框坐标列表，格式为[[x1,y1,x2,y2], ...]，像素坐标"
                },
                "logits": {
                    "type": "list",
                    "description": "每个检测框的置信度分数"
                },
                "phrases": {
                    "type": "list",
                    "description": "每个检测框对应的文本标签"
                },
                "size": {
                    "type": "list",
                    "description": "图像尺寸[高度, 宽度]"
                },
                "num_detections": {
                    "type": "int",
                    "description": "检测到的对象数量"
                },
                "mode": {
                    "type": "string",
                    "description": "运行模式（cpu/cuda）"
                }
            }
        }
    
    def visualize_results(self, result: Dict[str, Any], save_path: str = None) -> Image.Image:
        """可视化检测结果"""
        from PIL import ImageDraw, ImageFont
        import random
        
        if self.current_image is None:
            raise ValueError("No image loaded")
            
        # 复制图像用于绘制
        vis_image = self.current_image.copy()
        draw = ImageDraw.Draw(vis_image)
        
        # 尝试加载字体
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # 为每个检测框绘制
        boxes = result.get("boxes", [])
        phrases = result.get("phrases", [])
        logits = result.get("logits", [])
        
        for i, (box, phrase, logit) in enumerate(zip(boxes, phrases, logits)):
            # 随机颜色
            color = tuple(random.randint(0, 255) for _ in range(3))
            
            # 绘制边界框
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # 绘制标签
            label = f"{phrase} ({logit:.2f})"
            # 获取文本大小
            if hasattr(font, 'getbbox'):
                bbox = draw.textbbox((x1, y1), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                text_width, text_height = draw.textsize(label, font=font)
            
            # 绘制文本背景
            draw.rectangle([x1, y1-text_height-4, x1+text_width+4, y1], fill=color)
            draw.text((x1+2, y1-text_height-2), label, fill="white", font=font)
        
        if save_path:
            vis_image.save(save_path)
            
        return vis_image