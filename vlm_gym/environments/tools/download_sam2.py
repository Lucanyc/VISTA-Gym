#!/usr/bin/env python3
"""
SAM2完整下载脚本
"""

import os
import subprocess
import sys
from pathlib import Path

# 设置目标路径
TARGET_DIR = "/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/vlm_gym/environments/tools"
os.makedirs(TARGET_DIR, exist_ok=True)
os.chdir(TARGET_DIR)

print("开始下载和安装SAM2...")

# 1. 安装依赖包
print("\n[1/3] 安装依赖包...")
subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "transformers", "huggingface-hub", "numpy", "pillow", "opencv-python"], check=True)

# 2. 克隆并安装SAM2
print("\n[2/3] 克隆SAM2仓库...")
if not os.path.exists("segment-anything-2"):
    subprocess.run(["git", "clone", "https://github.com/facebookresearch/segment-anything-2.git"], check=True)

print("安装SAM2包...")
os.chdir("segment-anything-2")
subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
os.chdir("..")

# 3. 下载模型
print("\n[3/3] 下载SAM2模型...")
download_script = """
from huggingface_hub import snapshot_download
import os

model_dir = os.path.join(os.getcwd(), "sam2_models")
os.makedirs(model_dir, exist_ok=True)

# 下载facebook/sam2-hiera-large模型
print("正在从Hugging Face下载模型...")
model_path = snapshot_download(
    repo_id="facebook/sam2-hiera-large",
    cache_dir=model_dir,
    local_dir=os.path.join(model_dir, "sam2-hiera-large"),
    local_dir_use_symlinks=False
)

print(f"模型已下载到: {model_path}")
"""

# 执行下载
subprocess.run([sys.executable, "-c", download_script], check=True)

print("\n✅ SAM2安装完成！")
print(f"安装位置: {TARGET_DIR}")
print("\n使用方法:")
print("""
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 加载模型
predictor = SAM2ImagePredictor.from_pretrained("/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/vlm_gym/environments/tools/sam2_models/sam2-hiera-large")

# 使用模型
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(your_image)
    masks, scores, logits = predictor.predict(your_prompts)
""")