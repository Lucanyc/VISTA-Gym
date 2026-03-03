#!/usr/bin/env python3
"""
简化的工具功能测试 - 专注于工具本身而不是模型
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from vlm_gym.agents.vlm_agent_with_tools import VLMAgentWithTools
from vlm_gym.tools.tool_manager import VisionToolManager
from PIL import Image, ImageDraw

print("工具功能测试")
print("=" * 50)

# 1. 直接测试工具管理器
print("\n1. 测试工具管理器")
tool_manager = VisionToolManager({
    "enable_ocr": True,
    "enable_sam": False
})
print(f"✓ 可用工具: {list(tool_manager.tools.keys())}")

# 2. 创建测试图像
print("\n2. 创建测试图像")
test_img = Image.new('RGB', (400, 200), color='white')
draw = ImageDraw.Draw(test_img)
draw.text((50, 50), "Test Text 123", fill='black')
draw.text((50, 100), "Sample OCR Test", fill='blue')
test_img.save('test_simple.jpg')
print("✓ 测试图像已创建")

# 3. 测试 OCR 工具
print("\n3. 测试 OCR 工具")
try:
    result = tool_manager.extract_text('test_simple.jpg')
    print(f"✓ OCR 结果: {result}")
except Exception as e:
    print(f"⚠ OCR 测试失败 (可能需要安装 easyocr): {e}")

# 4. 测试图像处理工具
print("\n4. 测试图像处理工具")
try:
    result = tool_manager.crop_and_zoom('test_simple.jpg', [40, 40, 200, 120], zoom_factor=2.0)
    print(f"✓ 图像处理结果: {result['status']}")
    if 'image' in result and result['image']:
        result['image'].save('test_zoomed.jpg')
        print("✓ 放大的图像已保存为 test_zoomed.jpg")
except Exception as e:
    print(f"✗ 图像处理测试失败: {e}")

# 5. 测试工具选择逻辑
print("\n5. 测试工具选择逻辑")
questions = [
    ("What text is shown?", "应该选择 OCR 工具"),
    ("Zoom in on the text", "应该选择缩放工具"),
    ("How many words are there?", "应该选择计数工具"),
    ("What color is the sky?", "不需要工具")
]

# 创建一个不加载模型的 agent（仅测试工具选择）
class MockVLMAgent:
    def __init__(self, config):
        pass
    def act(self, observation):
        return "Mock response", {}

# 临时替换父类
import vlm_gym.agents.vlm_agent
original_vlm_agent = vlm_gym.agents.vlm_agent.VLMAgent
vlm_gym.agents.vlm_agent.VLMAgent = MockVLMAgent

try:
    agent = VLMAgentWithTools({
        "model_name": "mock",
        "enable_tools": True,
        "tool_selection_strategy": "adaptive"
    })
    
    for question, expected in questions:
        observation = {
            "image_path": "test_simple.jpg",
            "question": question
        }
        needs_tools, tool_plan = agent._analyze_task_requirements(observation)
        print(f"\n问题: {question}")
        print(f"需要工具: {needs_tools}, 计划: {tool_plan}")
        print(f"期望: {expected}")
finally:
    # 恢复原始类
    vlm_gym.agents.vlm_agent.VLMAgent = original_vlm_agent

# 清理
import os
for f in ['test_simple.jpg', 'test_zoomed.jpg']:
    if os.path.exists(f):
        os.remove(f)

print("\n" + "=" * 50)
print("测试完成！")
