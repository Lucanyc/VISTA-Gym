#!/usr/bin/env python3
"""
测试DeepEyes工具在VLMGym环境中的集成
"""

import sys
import os
sys.path.append('/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista')

import json
from PIL import Image
from vlm_gym.environments.tools import ToolBase, DeepEyesTool


def test_tool_registry():
    """测试工具注册"""
    print("=== Testing Tool Registry ===\n")
    
    # 1. 检查DeepEyes是否在注册表中
    print("1. Checking tool registry...")
    print(f"   Registered tools: {list(ToolBase.registry.keys())}")
    
    if 'deepeyes' in ToolBase.registry:
        print("   ✓ DeepEyes is registered")
    else:
        print("   ✗ DeepEyes is NOT registered")
        return False
    
    # 2. 测试通过registry创建工具
    print("\n2. Creating tool from registry...")
    try:
        tool = ToolBase.create('deepeyes')
        print(f"   ✓ Tool created: {tool.name}")
    except Exception as e:
        print(f"   ✗ Failed to create tool: {e}")
        return False
    
    return True


def test_deepeyes_in_vlmgym_context():
    """测试DeepEyes在VLMGym上下文中的使用"""
    print("\n=== Testing DeepEyes in VLMGym Context ===\n")
    
    # 创建工具实例 - 使用正确的初始化方式
    tool = DeepEyesTool()
    
    # 加载测试图像
    test_image_path = "/data/wang/meng/GYM-Work/dataset/ScienceQA/images/train/train_000000.png"
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        return
    
    image = Image.open(test_image_path)
    tool.reset(image)
    
    # 模拟VLMGym环境中的工具调用
    print("Simulating VLMGym tool usage...")
    
    # 1. Agent生成工具调用
    agent_output = """Let me examine the northern part of the map to determine which state is farthest north.
    
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [0, 0, 750, 200]}}
</tool_call>"""
    
    print(f"Agent output:\n{agent_output}\n")
    
    # 2. 执行工具调用
    result = tool.execute(agent_output)
    
    if result.get('success'):
        print("Tool execution successful!")
        print(f"- Tool used: {result.get('tool_used')}")
        print(f"- New image size: {result.get('new_size')}")
        print(f"- Processed output: {result.get('processed_output')[:100]}...")
    else:
        print(f"Tool execution failed: {result.get('error')}")
    
    # 3. Agent基于工具结果给出答案
    agent_final_output = """Based on the zoomed view of the northern part of the map, I can clearly see that West Virginia is positioned furthest north among the given options.

<answer>West Virginia</answer>"""
    
    print(f"\nAgent final output:\n{agent_final_output}\n")
    
    # 4. 执行最终答案提取
    final_result = tool.execute(agent_final_output)
    
    if final_result.get('success'):
        print("Answer extraction successful!")
        print(f"- Final answer: {final_result.get('final_answer')}")
        print(f"- Tool reported this was the final answer")
    else:
        print(f"Answer extraction failed: {final_result.get('error')}")


def test_tool_capabilities():
    """测试工具能力描述"""
    print("\n=== Testing Tool Capabilities ===\n")
    
    tool = DeepEyesTool()
    capabilities = tool.get_capabilities()
    
    print(json.dumps(capabilities, indent=2, ensure_ascii=False))


def test_vlmgym_environment_integration():
    """测试与VisionQAEnv的集成"""
    print("\n=== Testing VisionQAEnv Integration ===\n")
    
    try:
        from vlm_gym.environments.vision_qa_env import VisionQAEnv
        print("1. VisionQAEnv import successful")
        
        # 检查环境是否能识别DeepEyes工具
        # 这部分需要根据实际的VisionQAEnv实现来调整
        
        # 模拟环境配置
        env_config = {
            "tools": {
                "grounding_dino": False,
                "chartmoe": False,
                "diagram_formalizer": False,
                "deepeyes": True  # 启用DeepEyes
            }
        }
        
        print(f"2. Environment config with DeepEyes: {json.dumps(env_config, indent=2)}")
        
    except ImportError as e:
        print(f"Could not import VisionQAEnv: {e}")
        print("This is expected if we haven't integrated with the environment yet")


if __name__ == "__main__":
    print("DeepEyes Integration Test Suite\n")
    
    # 运行测试
    if test_tool_registry():
        print("\n✅ Tool registry test passed")
    else:
        print("\n❌ Tool registry test failed")
        sys.exit(1)
    
    test_deepeyes_in_vlmgym_context()
    test_tool_capabilities()
    test_vlmgym_environment_integration()
    
    print("\n=== Integration test completed ===")