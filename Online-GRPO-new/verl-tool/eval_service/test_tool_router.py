"""
工具路由代理服务器
将不同工具的请求路由到不同端口
"""
import asyncio
import aiohttp
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import json
import re
import base64

app = FastAPI(title="Tool Router")

# 工具到端口的映射
TOOL_PORTS = {
    "chartmoe": 5557,      # ChartMoE 独立运行
    "easyocr": 5558,       # EasyOCR 独立运行  
    "groundingdino": 5569, # GroundingDINO 独立运行
    "diagramformalizer": 5560,  # DiagramFormalizer 独立运行（旧环境）
    "multimath" : 5581,
    "finish": 5557,  # finish工具可以在任何端口
}

# 默认端口（如果检测不到工具）
DEFAULT_PORT = 6557

def extract_tool_from_action(action: str) -> str:
    """从action中提取工具名称"""
    # 尝试解析 <tool_call> 格式
    tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
    matches = re.findall(tool_call_pattern, action, re.DOTALL)
    
    if matches:
        try:
            params = json.loads(matches[0].strip())
            return params.get('tool', None)
        except:
            pass
    
    # 检查是否包含工具关键词
    for tool in TOOL_PORTS.keys():
        if tool in action.lower():
            return tool
    
    return None

def analyze_image_data(data):
    """分析数据中的图片信息"""
    image_info = {
        "has_image": False,
        "location": None,
        "format": None,
        "size": None
    }
    
    # 检查可能包含图片的所有位置
    potential_locations = [
        ("extra_fields", data.get("extra_fields", [])),
        ("images", data.get("images", [])),
        ("image", data.get("image", None)),
    ]
    
    for location_name, location_data in potential_locations:
        if location_data:
            if location_name == "extra_fields" and isinstance(location_data, list) and location_data:
                first_extra = location_data[0]
                if isinstance(first_extra, dict):
                    # 检查extra_fields中的图片键
                    for img_key in ['image', 'images', 'img', 'image_data', 'image_base64', 'pixel_values']:
                        if img_key in first_extra:
                            image_info["has_image"] = True
                            image_info["location"] = f"extra_fields[0]['{img_key}']"
                            img_data = first_extra[img_key]
                            if isinstance(img_data, str):
                                image_info["format"] = "base64" if img_data.startswith('data:image') or len(img_data) > 1000 else "path/url"
                                image_info["size"] = len(img_data)
                            elif isinstance(img_data, list):
                                image_info["format"] = "list"
                                image_info["size"] = len(img_data)
                            break
            elif location_name in ["images", "image"]:
                if location_data:
                    image_info["has_image"] = True
                    image_info["location"] = location_name
                    if isinstance(location_data, str):
                        image_info["format"] = "base64" if len(location_data) > 1000 else "path/url"
                        image_info["size"] = len(location_data)
                    elif isinstance(location_data, list):
                        image_info["format"] = "list"
                        image_info["size"] = len(location_data)
    
    return image_info

@app.post("/get_observation")
async def route_observation(request: Request):
    """路由工具请求到对应端口"""
    data = await request.json()
    
    # ===== 详细调试日志 =====
    print("=" * 70)
    print("🔍 ROUTER DEBUG - Incoming Request Analysis")
    print("=" * 70)
    
    # 基本信息
    print(f"📊 Request Overview:")
    print(f"  - Available keys: {list(data.keys())}")
    print(f"  - Number of trajectory_ids: {len(data.get('trajectory_ids', []))}")
    print(f"  - Number of actions: {len(data.get('actions', []))}")
    
    # 分析图片数据
    print(f"\n🖼️ Image Data Analysis:")
    image_info = analyze_image_data(data)
    if image_info["has_image"]:
        print(f"  ✅ Image data found!")
        print(f"  - Location: {image_info['location']}")
        print(f"  - Format: {image_info['format']}")
        print(f"  - Size: {image_info['size']} {'bytes' if image_info['format'] in ['base64', 'path/url'] else 'items'}")
    else:
        print(f"  ❌ No image data found in request")
    
    # extra_fields 详细分析
    print(f"\n📦 Extra Fields Analysis:")
    if "extra_fields" in data:
        extra_fields = data["extra_fields"]
        if extra_fields:
            print(f"  - Number of extra_fields: {len(extra_fields)}")
            if isinstance(extra_fields[0], dict):
                first_extra = extra_fields[0]
                print(f"  - Keys in first extra_field: {list(first_extra.keys())}")
                
                # 检查每个键的数据类型和大小
                for key, value in first_extra.items():
                    if value is None:
                        print(f"    • {key}: None")
                    elif isinstance(value, str):
                        if len(value) > 100:
                            print(f"    • {key}: string (length: {len(value)})")
                            if len(value) > 1000:
                                print(f"      → Likely base64 encoded data")
                        else:
                            print(f"    • {key}: string = '{value[:50]}{'...' if len(value) > 50 else ''}'")
                    elif isinstance(value, list):
                        print(f"    • {key}: list (length: {len(value)})")
                    elif isinstance(value, dict):
                        print(f"    • {key}: dict (keys: {list(value.keys())[:5]})")
                    else:
                        print(f"    • {key}: {type(value).__name__}")
            else:
                print(f"  - First extra_field type: {type(extra_fields[0]).__name__}")
        else:
            print(f"  - extra_fields is empty list")
    else:
        print(f"  - No extra_fields key in request")
    
    # Actions 分析
    print(f"\n🎯 Actions Analysis:")
    actions = data.get("actions", [])
    if actions:
        first_action = actions[0]
        print(f"  - First action preview: {first_action[:200]}{'...' if len(first_action) > 200 else ''}")
        
        # 提取工具信息
        tool = extract_tool_from_action(first_action)
        if tool:
            print(f"  - Detected tool: {tool}")
            print(f"  - Will route to port: {TOOL_PORTS.get(tool, DEFAULT_PORT)}")
        else:
            print(f"  - No tool detected, using default port: {DEFAULT_PORT}")
        
        # 检查action中是否有图片引用
        if "image" in first_action.lower() or "img" in first_action.lower():
            print(f"  - ⚠️ Action contains 'image' keyword but may not have actual image data")
    
    # 其他字段
    print(f"\n📋 Other Fields:")
    print(f"  - finish: {data.get('finish', 'Not present')}")
    print(f"  - is_last_step: {data.get('is_last_step', 'Not present')}")
    
    print("=" * 70 + "\n")
    # ===== 调试日志结束 =====
    
    # 分析每个action，按工具分组
    actions = data.get("actions", [])
    tool_groups = {}
    
    for i, action in enumerate(actions):
        # 检查是否是finish动作
        if data.get("finish", [])[i]:
            tool = "finish"
        else:
            tool = extract_tool_from_action(action)
        
        if tool and tool in TOOL_PORTS:
            port = TOOL_PORTS[tool]
        else:
            port = DEFAULT_PORT
            
        if port not in tool_groups:
            tool_groups[port] = {"indices": [], "data": {
                "trajectory_ids": [],
                "actions": [],
                "finish": [],
            }}
        
        tool_groups[port]["indices"].append(i)
        tool_groups[port]["data"]["trajectory_ids"].append(data["trajectory_ids"][i])
        tool_groups[port]["data"]["actions"].append(action)
        tool_groups[port]["data"]["finish"].append(data.get("finish", [])[i] if "finish" in data else False)
        
        # 传递额外字段
        if "extra_fields" in data:
            if "extra_fields" not in tool_groups[port]["data"]:
                tool_groups[port]["data"]["extra_fields"] = []
            tool_groups[port]["data"]["extra_fields"].append(data["extra_fields"][i])
        
        if "is_last_step" in data:
            if "is_last_step" not in tool_groups[port]["data"]:
                tool_groups[port]["data"]["is_last_step"] = []
            tool_groups[port]["data"]["is_last_step"].append(data["is_last_step"][i])
    
    # 调试：显示路由分组
    print(f"🚀 Routing to {len(tool_groups)} different ports:")
    for port, group_info in tool_groups.items():
        print(f"  - Port {port}: {len(group_info['indices'])} requests")
    
    # 并发调用不同端口
    results = {"observations": [None] * len(actions), 
               "dones": [False] * len(actions),
               "valids": [False] * len(actions)}
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for port, group_info in tool_groups.items():
            url = f"http://localhost:{port}/get_observation"
            tasks.append((port, group_info["indices"], 
                         session.post(url, json=group_info["data"])))
        
        # 执行所有请求
        for port, indices, task in tasks:
            try:
                async with task as response:
                    if response.status == 200:
                        resp_data = await response.json()
                        # 将结果放回正确位置
                        for j, orig_idx in enumerate(indices):
                            results["observations"][orig_idx] = resp_data["observations"][j]
                            results["dones"][orig_idx] = resp_data["dones"][j]
                            results["valids"][orig_idx] = resp_data["valids"][j]
                            
                        # 调试：显示工具响应
                        if resp_data["observations"]:
                            first_obs = resp_data["observations"][0]
                            if isinstance(first_obs, dict) and "error" in str(first_obs).lower():
                                print(f"  ⚠️ Port {port} returned error: {first_obs}")
                    else:
                        print(f"  ❌ Error from port {port}: {response.status}")
                        error_text = await response.text()
                        print(f"     Response: {error_text[:200]}")
                        # 对失败的请求返回默认值
                        for orig_idx in indices:
                            results["observations"][orig_idx] = f"Error: Tool server on port {port} returned {response.status}"
                            results["dones"][orig_idx] = True
                            results["valids"][orig_idx] = False
            except Exception as e:
                print(f"  ❌ Exception calling port {port}: {e}")
                for orig_idx in indices:
                    results["observations"][orig_idx] = f"Error: Failed to connect to port {port}: {str(e)}"
                    results["dones"][orig_idx] = True
                    results["valids"][orig_idx] = False
    
    print(f"✅ Request processed, returning {len(results['observations'])} observations\n")
    
    return JSONResponse(content=results)

@app.get("/health")
async def health_check():
    """健康检查端点"""
    health_status = {
        "status": "healthy",
        "role": "router",
        "tool_ports": TOOL_PORTS,
        "debug_mode": "enabled"
    }
    print(f"Health check requested: {health_status}")
    return health_status

if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("🚀 Starting Tool Router with Debug Mode")
    print(f"📍 Listening on: http://0.0.0.0:5556")
    print(f"🔧 Tool Ports: {TOOL_PORTS}")
    print("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=5556, log_level="info")