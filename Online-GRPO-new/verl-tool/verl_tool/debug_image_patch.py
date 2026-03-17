
import os
import torch
from PIL import Image

# 全局调试标志
DEBUG_IMAGE = True

def debug_log(msg):
    if DEBUG_IMAGE:
        print(f"[IMAGE_DEBUG] {msg}", flush=True)

# Hook数据加载
original_getitem = None

def patched_getitem(self, idx):
    data = original_getitem(self, idx)
    
    # 检查数据内容
    if DEBUG_IMAGE:
        debug_log(f"Dataset[{idx}] keys: {data.keys() if isinstance(data, dict) else type(data)}")
        
        if isinstance(data, dict):
            # 检查image_path
            if 'image_path' in data:
                debug_log(f"  image_path: {data['image_path']}")
                if os.path.exists(data['image_path']):
                    debug_log(f"  image_path exists: YES")
                else:
                    debug_log(f"  image_path exists: NO")
            
            # 检查是否有图像数据
            if 'image' in data:
                img = data['image']
                if isinstance(img, Image.Image):
                    debug_log(f"  image: PIL.Image {img.size}")
                elif isinstance(img, torch.Tensor):
                    debug_log(f"  image: Tensor {img.shape}")
                else:
                    debug_log(f"  image: {type(img)}")
            
            # 检查prompt
            if 'prompt' in data:
                prompt = str(data['prompt'])[:200]
                has_pad = '<|image_pad|>' in prompt
                debug_log(f"  prompt has <|image_pad|>: {has_pad}")
    
    return data

# Hook VLLM调用
original_generate = None

def patched_generate(self, *args, **kwargs):
    if DEBUG_IMAGE:
        debug_log("=== VLLM Generate Called ===")
        
        # 检查输入格式
        if args:
            first_arg = args[0]
            if isinstance(first_arg, list):
                debug_log(f"  Input type: list of {len(first_arg)} items")
                if first_arg:
                    item = first_arg[0]
                    if isinstance(item, dict):
                        debug_log(f"    First item keys: {item.keys()}")
                        if 'multi_modal_data' in item:
                            mm = item['multi_modal_data']
                            debug_log(f"    multi_modal_data keys: {mm.keys() if isinstance(mm, dict) else type(mm)}")
                            if isinstance(mm, dict) and 'image' in mm:
                                img = mm['image']
                                if isinstance(img, list):
                                    debug_log(f"      image: list of {len(img)}")
                                    if img:
                                        debug_log(f"        first: {type(img[0])}")
                                elif img is None:
                                    debug_log(f"      image: None")
                                else:
                                    debug_log(f"      image: {type(img)}")
                    elif isinstance(item, str):
                        has_pad = '<|image_pad|>' in item
                        debug_log(f"    String input, has <|image_pad|>: {has_pad}")
            elif isinstance(first_arg, str):
                has_pad = '<|image_pad|>' in first_arg
                debug_log(f"  String input, has <|image_pad|>: {has_pad}")
        
        # 检查kwargs中的images
        if 'images' in kwargs:
            imgs = kwargs['images']
            if imgs is None:
                debug_log(f"  kwargs['images']: None")
            elif isinstance(imgs, list):
                debug_log(f"  kwargs['images']: list of {len(imgs)}")
                if imgs:
                    debug_log(f"    first: {type(imgs[0])}")
            else:
                debug_log(f"  kwargs['images']: {type(imgs)}")
    
    return original_generate(self, *args, **kwargs)

# 应用补丁
def apply_debug_patches():
    print("[IMAGE_DEBUG] Applying debug patches...", flush=True)
    
    # Patch数据集
    try:
        from verl.utils.dataset import RLHFDataset
        global original_getitem
        original_getitem = RLHFDataset.__getitem__
        RLHFDataset.__getitem__ = patched_getitem
        print("[IMAGE_DEBUG] Patched RLHFDataset", flush=True)
    except Exception as e:
        print(f"[IMAGE_DEBUG] Failed to patch dataset: {e}", flush=True)
    
    # Patch VLLM
    try:
        from vllm import LLM
        global original_generate
        original_generate = LLM.generate
        LLM.generate = patched_generate
        print("[IMAGE_DEBUG] Patched VLLM.generate", flush=True)
    except Exception as e:
        print(f"[IMAGE_DEBUG] Failed to patch VLLM: {e}", flush=True)

# 自动应用
apply_debug_patches()
