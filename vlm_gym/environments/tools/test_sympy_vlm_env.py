# test_sympy_in_vlm_env.py
"""
测试SymPy几何工具在VLM环境中的实际使用
"""
import sys
sys.path.append('/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista')

import json
import logging
from vlm_gym.environments.vision_qa_env import VisionQAEnv

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sympy_tool_in_env():
    """在VLM环境中测试SymPy几何工具"""
    
    print("=== 在VLM环境中测试SymPy几何工具 ===\n")
    
    try:
        # 1. 创建环境并启用工具
        print("1. 初始化VLM环境...")
        env_config = {
            "enable_tools": True,  # 启用工具
            "tools": ["sympy_geometry"],  # 指定使用的工具
            "tool_configs": {
                "sympy_geometry": {
                    "precision": 4  # 设置精度
                }
            }
        }
        
        # 创建环境 - 使用一个简单的任务测试
        env = VisionQAEnv(
            task_name="mathvista",  # 或其他包含几何问题的任务
            config=env_config
        )
        
        print("✅ 环境创建成功\n")
        
        # 2. 重置环境获取初始观察
        print("2. 重置环境...")
        observation = env.reset()
        print(f"初始观察:")
        print(f"  - 问题: {observation.get('question', 'N/A')}")
        print(f"  - 图像路径: {observation.get('image_path', 'N/A')}")
        print(f"  - 可用工具: {env.available_tools if hasattr(env, 'available_tools') else 'N/A'}")
        print()
        
        # 3. 测试工具调用
        print("3. 测试SymPy几何工具调用...")
        
        # 测试用例1: 计算三角形角度
        print("\n测试用例1: 计算三角形角度")
        action = {
            "tool": "sympy_geometry",
            "params": json.dumps({
                "function": "triangle_angle",
                "args": {
                    "A": [0, 0],
                    "B": [3, 0], 
                    "C": [0, 4],
                    "vertex": "A"
                }
            })
        }
        
        print(f"发送动作: {json.dumps(action, indent=2)}")
        
        # 执行动作
        result = env.step(action)
        observation, reward, done, info = result
        
        print(f"\n环境响应:")
        print(f"  - 观察: {observation}")
        print(f"  - 奖励: {reward}")
        print(f"  - 完成: {done}")
        print(f"  - 信息: {info}")
        
        # 测试用例2: 勾股定理计算
        print("\n\n测试用例2: 勾股定理计算")
        action2 = {
            "tool": "sympy_geometry",
            "params": json.dumps({
                "function": "pythagorean",
                "args": {
                    "a": 6,
                    "b": 8
                }
            })
        }
        
        print(f"发送动作: {json.dumps(action2, indent=2)}")
        
        result2 = env.step(action2)
        observation2, reward2, done2, info2 = result2
        
        print(f"\n环境响应:")
        print(f"  - 工具输出: {observation2.get('tool_output', 'N/A')}")
        
        # 测试用例3: 多边形面积
        print("\n\n测试用例3: 计算五边形面积")
        action3 = {
            "tool": "sympy_geometry",
            "params": json.dumps({
                "function": "polygon_area",
                "args": {
                    "vertices": [
                        [0, 0],
                        [2, 0],
                        [3, 1.5],
                        [1, 2.5],
                        [-1, 1]
                    ]
                }
            })
        }
        
        print(f"发送动作: {json.dumps(action3, indent=2)}")
        
        result3 = env.step(action3)
        observation3, reward3, done3, info3 = result3
        
        print(f"\n环境响应:")
        print(f"  - 工具输出: {observation3.get('tool_output', 'N/A')}")
        
        # 4. 测试错误处理
        print("\n\n4. 测试错误处理...")
        error_action = {
            "tool": "sympy_geometry",
            "params": json.dumps({
                "function": "invalid_function",
                "args": {}
            })
        }
        
        print(f"发送错误动作: {json.dumps(error_action, indent=2)}")
        
        error_result = env.step(error_action)
        error_obs, _, _, error_info = error_result
        
        print(f"\n错误处理响应:")
        print(f"  - 错误信息: {error_obs.get('error', 'N/A')}")
        print(f"  - 可用函数: {error_obs.get('available_functions', 'N/A')}")
        
        # 5. 显示环境状态
        print("\n\n5. 环境状态总结:")
        if hasattr(env, 'get_state'):
            state = env.get_state()
            print(f"  - 当前状态: {state}")
        
        if hasattr(env, 'tools'):
            print(f"  - 已加载工具: {list(env.tools.keys())}")
            
        print("\n✅ 测试完成!")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理环境
        if 'env' in locals():
            env.close()
            print("\n环境已关闭")

def test_tool_with_mock_env():
    """使用模拟环境测试工具（如果真实环境有问题）"""
    print("\n=== 模拟环境测试 ===\n")
    
    from vlm_gym.environments.tools.geometry_tools.sympy_geometry import SympyGeometryTool
    
    # 创建工具实例
    tool = SympyGeometryTool()
    tool.reset()
    
    # 模拟环境调用工具的过程
    test_cases = [
        {
            "name": "三角形内角和验证",
            "action": {
                "function": "triangle_angle_from_two",
                "args": {
                    "angle1": 60,
                    "angle2": 80
                }
            }
        },
        {
            "name": "等边三角形检测",
            "action": {
                "function": "triangle_type",
                "args": {
                    "A": [0, 0],
                    "B": [1, 0],
                    "C": [0.5, 0.866]  # 约等于 sqrt(3)/2
                }
            }
        },
        {
            "name": "点到直线距离",
            "action": {
                "function": "distance_point_to_line",
                "args": {
                    "point": [2, 3],
                    "line_p1": [0, 0],
                    "line_p2": [1, 1]
                }
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"测试: {test_case['name']}")
        result = tool.execute(json.dumps(test_case['action']))
        print(f"结果: {result.get('formatted', result)}\n")

if __name__ == "__main__":
    # 先尝试在真实环境中测试
    test_sympy_tool_in_env()
    
    # 如果真实环境有问题，可以使用模拟测试
    # test_tool_with_mock_env()