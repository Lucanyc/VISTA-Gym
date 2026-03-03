# test_import.py
import sys
sys.path.append('/data/wang/meng/GYM-Work/vlm_gym-tool-usage-geometry')

try:
    from vlm_gym.environments.tools.geometry_tools import DiagramFormalizerTool
    print("✅ 从geometry_tools导入成功")
except Exception as e:
    print(f"❌ 导入失败: {e}")

try:
    from vlm_gym.environments.tools import DiagramFormalizerTool
    print("✅ 从tools导入成功")
except Exception as e:
    print(f"❌ 导入失败: {e}")

# 测试创建实例
try:
    tool = DiagramFormalizerTool()
    print("✅ 工具实例创建成功")
except Exception as e:
    print(f"❌ 创建失败: {e}")