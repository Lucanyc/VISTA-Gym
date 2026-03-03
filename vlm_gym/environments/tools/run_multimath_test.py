import warnings
warnings.filterwarnings("ignore")

# 猴子补丁修复 torch.load 检查
import torch
original_load = torch.load

def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)

torch.load = patched_load

# 现在运行你的测试
import subprocess
subprocess.run([
    "python", "test_multimath_geometry3k.py",
    "--model_path", "./checkpoints/multimath-7b-llava-v1.5",
    "--single_problem", "0",
    "--debug"
])