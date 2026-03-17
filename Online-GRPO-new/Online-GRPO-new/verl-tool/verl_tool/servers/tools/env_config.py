"""
环境配置 - 所有工具共享
必须在其他 import 之前导入
"""
import os

NEW_STORAGE_PATH = "/projects/burg/menglu/acl_training"

# Ray 临时目录
os.environ['RAY_TMPDIR'] = f"{NEW_STORAGE_PATH}/ray"

# 通用临时目录
os.environ['TMPDIR'] = f"{NEW_STORAGE_PATH}/tmp"
os.environ['TEMP'] = f"{NEW_STORAGE_PATH}/tmp"
os.environ['TMP'] = f"{NEW_STORAGE_PATH}/tmp"

# PyTorch 缓存
os.environ['TORCH_HOME'] = f"{NEW_STORAGE_PATH}/torch_cache"
os.environ['TORCHINDUCTOR_CACHE_DIR'] = f"{NEW_STORAGE_PATH}/torch_cache/inductor"

# Triton 缓存
os.environ['TRITON_CACHE_DIR'] = f"{NEW_STORAGE_PATH}/triton_cache"

# HuggingFace 缓存
os.environ['HF_HOME'] = f"{NEW_STORAGE_PATH}/huggingface"
os.environ['HF_MODULES_CACHE'] = f"{NEW_STORAGE_PATH}/huggingface/modules"
os.environ['TRANSFORMERS_CACHE'] = f"{NEW_STORAGE_PATH}/huggingface"
os.environ['HUGGINGFACE_HUB_CACHE'] = f"{NEW_STORAGE_PATH}/huggingface/hub"

# XDG 缓存
os.environ['XDG_CACHE_HOME'] = f"{NEW_STORAGE_PATH}/cache"

# 创建目录
for path in [
    f"{NEW_STORAGE_PATH}/ray",
    f"{NEW_STORAGE_PATH}/tmp",
    f"{NEW_STORAGE_PATH}/torch_cache",
    f"{NEW_STORAGE_PATH}/triton_cache",
    f"{NEW_STORAGE_PATH}/cache",
    f"{NEW_STORAGE_PATH}/huggingface",
    f"{NEW_STORAGE_PATH}/huggingface/modules",
]:
    os.makedirs(path, exist_ok=True)

print(f"[ENV] Storage configured: {NEW_STORAGE_PATH}")