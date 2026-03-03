# test_import.py
import sys
sys.path.append('/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista')

# 测试导入
from vlm_gym.environments.tools import SympyGeometryTool

# 验证工具注册
from vlm_gym.environments.tools.base import ToolBase

print("已注册的工具:")
for name, tool_class in ToolBase.registry.items():
    print(f"  - {name}: {tool_class}")

# 验证 sympy_geometry 是否在其中
if "sympy_geometry" in ToolBase.registry:
    print("\n✅ SympyGeometryTool 已成功注册!")
else:
    print("\n❌ SympyGeometryTool 未注册")

# 测试创建实例
try:
    tool = SympyGeometryTool()
    print(f"✅ 工具实例创建成功: {tool.name}")
except Exception as e:
    print(f"❌ 创建失败: {e}")