#!/usr/bin/env python3
"""
测试生产服务（5557等端口）
"""
import json
import requests
import time

# 生产服务端口
PRODUCTION_PORTS = {
    "chartmoe": 5557,
    "easyocr": 5558,
    "groundingdino": 5569,
    "diagramformalizer": 5560,
    "multimath": 5581,
}

ROUTER_URL = "http://localhost:5556"

# 测试图片
TEST_IMAGES = {
    "chart": "/mnt/nfs/meng/Online-GRPO-new/test_datasets/chartqa/test/png/41699051005347.png",
    "geometry": "/mnt/nfs/meng/Online-GRPO-new/test_datasets/UniGeo/images/0.png",
}

def test_direct_production_services():
    """直接测试生产服务"""
    print("\n" + "="*60)
    print("测试生产服务（5557等端口）")
    print("="*60)
    
    # 测试ChartMoE
    print("\n[ChartMoE - 5557]")
    data = {
        "trajectory_ids": ["test_1"],
        "actions": ['<tool_call>{"tool": "chartmoe", "task": "to_table"}</tool_call>'],
        "finish": [False],
        "is_last_step": [False],  # 生产服务需要这个字段
        "extra_fields": [{"images": [TEST_IMAGES["chart"]]}]
    }
    
    try:
        response = requests.post(f"http://localhost:5557/get_observation", json=data, timeout=30)
        print(f"状态: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Valid: {result['valids'][0]}")
            obs = result['observations'][0]
            print(f"Observation: {str(obs)[:200]}...")
    except Exception as e:
        print(f"错误: {e}")
    
    # 测试EasyOCR
    print("\n[EasyOCR - 5558]")
    data["actions"] = ['<tool_call>{"tool": "easyocr", "task": "ocr", "languages": ["en"]}</tool_call>']
    
    try:
        response = requests.post(f"http://localhost:5558/get_observation", json=data, timeout=30)
        print(f"状态: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Valid: {result['valids'][0]}")
            obs = result['observations'][0]
            print(f"Observation: {str(obs)[:200]}...")
    except Exception as e:
        print(f"错误: {e}")
    
    # 测试GroundingDINO
    print("\n[GroundingDINO - 5569]")
    data["actions"] = ['<tool_call>{"tool": "groundingdino", "task": "detect", "query": "objects"}</tool_call>']
    data["extra_fields"] = [{"images": [TEST_IMAGES["geometry"]]}]
    
    try:
        response = requests.post(f"http://localhost:5569/get_observation", json=data, timeout=30)
        print(f"状态: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Valid: {result['valids'][0]}")
            obs = result['observations'][0]
            print(f"Observation: {str(obs)[:200]}...")
    except Exception as e:
        print(f"错误: {e}")
    
    # 测试MultiMath
    print("\n[MultiMath - 5581]")
    data["actions"] = ['<tool_call>{"tool": "multimath", "task": "solve", "question": "What is 25 + 37?"}</tool_call>']
    
    try:
        response = requests.post(f"http://localhost:5581/get_observation", json=data, timeout=30)
        print(f"状态: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Valid: {result['valids'][0]}")
            obs = result['observations'][0]
            print(f"Observation: {str(obs)[:200]}...")
    except Exception as e:
        print(f"错误: {e}")

def test_router_with_production():
    """测试Router路由到生产服务"""
    print("\n" + "="*60)
    print("测试Router路由到生产服务")
    print("="*60)
    
    # 混合请求
    data = {
        "trajectory_ids": ["router_1", "router_2", "router_3"],
        "actions": [
            '<tool_call>{"tool": "chartmoe", "task": "to_table"}</tool_call>',
            '<tool_call>{"tool": "easyocr", "task": "ocr", "languages": ["en"]}</tool_call>',
            '<tool_call>{"tool": "multimath", "task": "solve", "question": "15 + 27 = ?"}</tool_call>'
        ],
        "finish": [False, False, False],
        "is_last_step": [False, False, False],  # 生产服务需要
        "extra_fields": [
            {"images": [TEST_IMAGES["chart"]]},
            {"images": [TEST_IMAGES["chart"]]},
            {"images": []}
        ]
    }
    
    try:
        start = time.time()
        response = requests.post(f"{ROUTER_URL}/get_observation", json=data, timeout=60)
        elapsed = time.time() - start
        
        print(f"\n状态: {response.status_code}")
        print(f"耗时: {elapsed:.2f}秒")
        
        if response.status_code == 200:
            result = response.json()
            tools = ["ChartMoE", "EasyOCR", "MultiMath"]
            for i, tool in enumerate(tools):
                valid = result['valids'][i]
                obs = result['observations'][i]
                if valid and obs and not str(obs).startswith("Error"):
                    print(f"✅ {tool}: Valid")
                else:
                    print(f"❌ {tool}: {str(obs)[:100] if obs else 'No response'}")
    except Exception as e:
        print(f"错误: {e}")

def main():
    print("\n生产服务测试脚本")
    print("="*60)
    
    # 检查Router配置
    print("\n[检查Router配置]")
    try:
        response = requests.get(f"{ROUTER_URL}/health")
        if response.status_code == 200:
            config = response.json()
            ports = config.get("tool_ports", {})
            print("Router当前配置的端口:")
            for tool, port in ports.items():
                print(f"  {tool}: {port}")
            
            # 检查是否指向生产端口
            if ports.get("chartmoe") == 5557:
                print("\n✅ Router已配置为使用生产端口")
            else:
                print("\n⚠️  Router仍指向测试端口，请修改配置并重启Router")
                print("   需要修改 test_tool_router.py 中的 TOOL_PORTS")
    except Exception as e:
        print(f"无法连接Router: {e}")
    
    # 测试服务
    test_direct_production_services()
    test_router_with_production()
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)

if __name__ == "__main__":
    main()