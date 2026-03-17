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
    "chartmoe": 6658,      # ChartMoE 独立运行
    "easyocr": 6758,       # EasyOCR 独立运行  
    "groundingdino": 6569, # GroundingDINO 独立运行
    "diagramformalizer": 7866,  # DiagramFormalizer 独立运行（旧环境）
    "multimath" : 6582,
    "gllava": 8690,
}

# 默认端口（如果检测不到工具）
DEFAULT_PORT = 6658  # 改为一个实际在线的端口作为fallback

# ==================== 工具端口检查 ====================

async def check_tool_status():
    """检查所有工具端口是否在线"""
    status = {}
    async with aiohttp.ClientSession() as session:
        for tool, port in TOOL_PORTS.items():
            try:
                async with session.get(
                    f"http://localhost:{port}/health", 
                    timeout=aiohttp.ClientTimeout(total=3)
                ) as resp:
                    status[tool] = {
                        "port": port, 
                        "status": "✅ online", 
                        "code": resp.status
                    }
            except aiohttp.ClientConnectorError:
                status[tool] = {
                    "port": port, 
                    "status": "❌ offline (connection refused)"
                }
            except asyncio.TimeoutError:
                status[tool] = {
                    "port": port, 
                    "status": "⚠️ timeout (3s)"
                }
            except Exception as e:
                status[tool] = {
                    "port": port, 
                    "status": f"❌ error: {str(e)}"
                }
    return status

@app.on_event("startup")
async def startup_check():
    """启动时自动检查所有工具端口"""
    print("\n" + "=" * 70)
    print("🔍 Startup Tool Health Check - Waiting 2s for services...")
    print("=" * 70)
    await asyncio.sleep(2)  # 等待其他服务启动
    
    status = await check_tool_status()
    online = sum(1 for v in status.values() if "online" in v["status"])
    
    print(f"\n📊 Tool Status Summary: {online}/{len(TOOL_PORTS)} online")
    print("-" * 50)
    for tool, info in status.items():
        print(f"  {info['status']}  {tool} (port {info['port']})")
    print("-" * 50)
    
    if online == len(TOOL_PORTS):
        print("🎉 All tools are online!")
    else:
        offline = len(TOOL_PORTS) - online
        print(f"⚠️ Warning: {offline} tool(s) are offline!")
    print("=" * 70 + "\n")

@app.get("/check_tools")
async def check_tools():
    """检查所有工具端口是否在线（HTTP端点）"""
    status = await check_tool_status()
    online = sum(1 for v in status.values() if "online" in v["status"])
    
    print(f"\n🔍 Tool Status Check: {online}/{len(TOOL_PORTS)} online")
    for tool, info in status.items():
        print(f"  {info['status']}  {tool} (port {info['port']})")
    
    return {
        "summary": f"{online}/{len(TOOL_PORTS)} online",
        "online_count": online,
        "total_count": len(TOOL_PORTS),
        "tools": status
    }

# ==================== 工具输出格式化 ====================

def format_tool_output(tool_name: str, port: int, raw_observation: any) -> any:
    """
    统一格式化工具输出，使其更适合7B模型学习
    - 添加工具标识前缀
    - EasyOCR: 保留bbox信息的简化格式
    - ChartMoE/MultiMath/G-LLaVA: 保留完整推理过程
    """
    
    # 处理错误情况
    if isinstance(raw_observation, str) and raw_observation.startswith("Error"):
        return raw_observation
    
    # ========== EasyOCR格式化 - 保留bbox信息 ==========
    if tool_name == "easyocr" or port == 5558:
        try:
            # 尝试解析JSON字符串
            if isinstance(raw_observation, str):
                try:
                    data = json.loads(raw_observation)
                except:
                    return f"EasyOCR Detection Result:\n{raw_observation}"
            elif isinstance(raw_observation, dict):
                data = raw_observation
            else:
                return f"EasyOCR Detection Result:\n{raw_observation}"
            
            # 提取文本和位置信息
            detected_items = []
            
            # 优先从detections字段提取（包含bbox和confidence信息）
            if 'detections' in data and isinstance(data['detections'], list) and data['detections']:
                for detection in data['detections']:
                    if 'text' in detection:
                        text = detection['text']
                        # 提取并简化bbox（4个点简化为2个点：左上和右下）
                        if 'bbox' in detection and isinstance(detection['bbox'], list) and len(detection['bbox']) >= 2:
                            try:
                                bbox = detection['bbox']
                                # 获取左上角和右下角坐标
                                x1, y1 = int(bbox[0][0]), int(bbox[0][1])
                                # 如果有4个点，使用第3个点作为右下角；否则使用第2个点
                                if len(bbox) >= 3:
                                    x2, y2 = int(bbox[2][0]), int(bbox[2][1])
                                else:
                                    x2, y2 = int(bbox[1][0]), int(bbox[1][1])
                                # 格式化为 "text [x1,y1,x2,y2]"
                                detected_items.append(f"{text} [{x1},{y1},{x2},{y2}]")
                            except Exception as e:
                                # 如果坐标解析失败，只添加文本
                                detected_items.append(text)
                        else:
                            # 没有有效的bbox，只添加文本
                            detected_items.append(text)
            
            # 如果detections为空但有all_texts字段（没有位置信息）
            elif 'all_texts' in data and data['all_texts']:
                detected_items = data['all_texts']
            
            # 如果有processed_output字段作为备选
            elif 'processed_output' in data:
                if data['processed_output']:
                    return f"EasyOCR Detection Result:\n{data['processed_output']}"
                else:
                    return "EasyOCR Detection Result:\nNo text detected in image"
            
            # 组合最终结果
            if detected_items:
                # 用 " | " 分隔各个检测项
                return "EasyOCR Detection Result:\n" + " | ".join(detected_items)
            elif 'num_detections' in data and data.get('num_detections', 0) == 0:
                return "EasyOCR Detection Result:\nNo text detected in image"
            else:
                # 兜底：返回原始内容
                return f"EasyOCR Detection Result:\n{raw_observation}"
            
        except Exception as e:
            print(f"Warning: Failed to format EasyOCR output: {e}")
            return f"EasyOCR Detection Result:\n{raw_observation}"
    
    # ========== ChartMoE - 保留完整内容包括表格 ==========
    elif tool_name == "chartmoe" or port == 6658:
        prefix = "ChartMoE Analysis Result:\n"
        
        if isinstance(raw_observation, dict):
            # 优先使用full_response字段
            if 'full_response' in raw_observation:
                return prefix + raw_observation['full_response']
            # 如果有table_data字段，确保表格被保留
            elif 'table_data' in raw_observation:
                return prefix + raw_observation['table_data']
            # 如果有output字段
            elif 'output' in raw_observation:
                return prefix + raw_observation['output']
            # 如果有processed_output字段
            elif 'processed_output' in raw_observation:
                return prefix + raw_observation['processed_output']
        
        # 如果是字符串或其他格式，直接添加前缀
        return prefix + str(raw_observation)
    
    # ========== MultiMath - 保留完整推理过程 ==========
    elif tool_name == "multimath" or port == 6582:
        prefix = "MultiMath Solution:\n"
        
        if isinstance(raw_observation, dict):
            # 优先返回full_response（包含完整推理步骤）
            if 'full_response' in raw_observation and raw_observation['full_response']:
                return prefix + raw_observation['full_response']
            # 其次尝试output字段
            elif 'output' in raw_observation and raw_observation['output']:
                return prefix + raw_observation['output']
            # 如果有answer和steps，组合它们
            elif 'answer' in raw_observation:
                result_parts = []
                if 'steps' in raw_observation and raw_observation['steps']:
                    result_parts.append(raw_observation['steps'])
                if 'reasoning' in raw_observation and raw_observation['reasoning']:
                    result_parts.append(f"Reasoning: {raw_observation['reasoning']}")
                result_parts.append(f"Answer: {raw_observation['answer']}")
                if 'final_answer' in raw_observation and raw_observation['final_answer'] != raw_observation['answer']:
                    result_parts.append(f"Final Answer: {raw_observation['final_answer']}")
                return prefix + "\n".join(result_parts)
            # 如果有result字段
            elif 'result' in raw_observation:
                return prefix + str(raw_observation['result'])
        
        # 如果是字符串或其他格式，直接添加前缀
        return prefix + str(raw_observation)
    
    # ========== G-LLaVA - 保留完整推理过程 ==========
    elif tool_name == "gllava" or tool_name == "g-llava":
        prefix = "G-LLaVA Geometric Analysis:\n"
        
        if isinstance(raw_observation, dict):
            # 优先返回response字段（包含完整推理）
            if 'response' in raw_observation and raw_observation['response']:
                return prefix + raw_observation['response']
            # 其次尝试full_response
            elif 'full_response' in raw_observation and raw_observation['full_response']:
                return prefix + raw_observation['full_response']
            # 如果有output字段
            elif 'output' in raw_observation and raw_observation['output']:
                return prefix + raw_observation['output']
            # 如果有result字段
            elif 'result' in raw_observation and raw_observation['result']:
                return prefix + raw_observation['result']
            # 组合其他可用信息
            else:
                output_parts = []
                if 'model' in raw_observation:
                    output_parts.append(f"Model: {raw_observation['model']}")
                if 'method' in raw_observation:
                    output_parts.append(f"Method: {raw_observation['method']}")
                if 'reasoning' in raw_observation:
                    output_parts.append(f"Reasoning: {raw_observation['reasoning']}")
                if 'answer' in raw_observation:
                    output_parts.append(f"Answer: {raw_observation['answer']}")
                
                if output_parts:
                    return prefix + "\n".join(output_parts)
        
        # 如果是字符串或其他格式，直接添加前缀
        return prefix + str(raw_observation)
    
    # ========== DiagramFormalizer - 保留完整内容 ==========
    elif tool_name == "diagramformalizer":
        prefix = "DiagramFormalizer Analysis:\n"
        
        if isinstance(raw_observation, dict):
            if 'full_response' in raw_observation:
                return prefix + raw_observation['full_response']
            elif 'result' in raw_observation:
                return prefix + raw_observation['result']
            elif 'output' in raw_observation:
                return prefix + raw_observation['output']
        
        return prefix + str(raw_observation)
    
    # ========== GroundingDINO - 格式化检测结果 ==========
    elif tool_name == "groundingdino":
        prefix = "GroundingDINO Object Detection Result:\n"
        
        if isinstance(raw_observation, dict):
            if 'detections' in raw_observation and isinstance(raw_observation['detections'], list):
                # 格式化检测结果
                results = []
                for det in raw_observation['detections']:
                    if 'label' in det:
                        label = det['label']
                        if 'bbox' in det:
                            bbox = det['bbox']
                            # 格式化为 "label [x1,y1,x2,y2]"
                            if isinstance(bbox, list) and len(bbox) >= 4:
                                results.append(f"{label} [{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]")
                            else:
                                results.append(label)
                        else:
                            results.append(label)
                
                if results:
                    return prefix + " | ".join(results)
                else:
                    return prefix + "No objects detected"
            
            elif 'num_detections' in raw_observation:
                num = raw_observation['num_detections']
                phrases = raw_observation.get('phrases', [])
                if num > 0 and phrases:
                    return prefix + f"Detected {num} objects: {', '.join(phrases)}"
                else:
                    return prefix + "No objects detected"
            
            elif 'result' in raw_observation:
                return prefix + str(raw_observation['result'])
            elif 'output' in raw_observation:
                return prefix + str(raw_observation['output'])
        
        return prefix + str(raw_observation)
    
    # ========== 默认：返回原始内容 ==========
    return raw_observation

# ==================== 工具调用修复 ====================

def fix_tool_call(action: str) -> str:
    """修复工具调用格式，识别关键字并返回正确格式"""
    # 检查是否包含<tool_call>标记
    if '<tool_call>' not in action:
        return action
    
    # 提取关键字（不区分大小写）
    content_lower = action.lower()
    
    # 根据关键字识别工具并返回正确格式
    if 'chartmoe' in content_lower or 'chart' in content_lower:
        return '<tool_call>{"tool": "chartmoe", "task": "to_table"}</tool_call>'
    
    elif 'groundingdino' in content_lower or 'grounding' in content_lower or 'dino' in content_lower:
        return '<tool_call>{"tool": "groundingdino", "task": "detect", "query": "objects"}</tool_call>'
    
    elif 'diagramformalizer' in content_lower or 'diagram' in content_lower or 'formalizer' in content_lower:
        return '<tool_call>{"tool": "diagramformalizer", "task": "analyze"}</tool_call>'
    
    elif 'easyocr' in content_lower or 'ocr' in content_lower:
        return '<tool_call>{"tool": "easyocr", "parameters": {"task": "detect_and_recognize"}}</tool_call>'
    
    elif 'multimath' in content_lower or 'math' in content_lower:
        return '<tool_call>{"tool": "multimath", "parameters": {"task": "solve", "question": "question"}}</tool_call>'
    
    elif 'gllava' in content_lower or 'g-llava' in content_lower or 'llava' in content_lower:
        return '<tool_call>{"tool": "gllava", "task": "solve", "question": "question"}</tool_call>'
    
    # 如果无法识别，返回原始内容
    return action

def extract_tool_from_action(action: str) -> str:
    """从action中提取工具名称"""
    # 尝试解析 <tool_call> 格式
    tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
    matches = re.findall(tool_call_pattern, action, re.DOTALL)
    
    if matches:
        try:
            params = json.loads(matches[0].strip())
            tool_name = params.get('tool', None)
            if tool_name:
                return tool_name.lower()  # 统一转换为小写
        except:
            pass  
    # 检查是否包含工具关键词
    for tool in TOOL_PORTS.keys():
        if tool in action.lower():
            return tool
    
    return None

# ==================== 图片数据分析 ====================

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

# ==================== 主路由 ====================

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
            print(f"  - No tool detected in action")
        
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
    finish_flags = data.get("finish", [False] * len(actions))
    tool_groups = {}
    
    # 初始化结果
    results = {"observations": [None] * len(actions), 
               "dones": [False] * len(actions),
               "valids": [False] * len(actions)}
    
    for i, action in enumerate(actions):
        # ===== 关键修复：finish=True 直接返回空observation，不路由到任何端口 =====
        if finish_flags[i]:
            results["observations"][i] = ""
            results["dones"][i] = True
            results["valids"][i] = True
            print(f"  ✅ Action {i}: finish=True, skipping tool routing")
            continue
        
        # 修复可能的格式错误
        fixed_action = fix_tool_call(action)
        
        # 如果格式被修复，打印日志
        if fixed_action != action:
            print(f"  🔧 Fixed tool call format for action {i}")
            print(f"     Original: {action[:100]}...")
            print(f"     Fixed: {fixed_action[:100]}...")
        
        # 提取工具名称
        tool = extract_tool_from_action(fixed_action)
        
        if tool and tool in TOOL_PORTS:
            port = TOOL_PORTS[tool]
        elif tool:
            # 工具名称识别到了但不在端口映射中
            print(f"  ⚠️ Action {i}: Unknown tool '{tool}', using default port {DEFAULT_PORT}")
            port = DEFAULT_PORT
        else:
            # 没有检测到任何工具调用
            print(f"  ⚠️ Action {i}: No tool detected in action, using default port {DEFAULT_PORT}")
            port = DEFAULT_PORT
            
        if port not in tool_groups:
            tool_groups[port] = {"indices": [], "data": {
                "trajectory_ids": [],
                "actions": [],
                "finish": [],
            }}
        
        tool_groups[port]["indices"].append(i)
        tool_groups[port]["data"]["trajectory_ids"].append(data["trajectory_ids"][i])
        tool_groups[port]["data"]["actions"].append(fixed_action)  # 使用修复后的action
        tool_groups[port]["data"]["finish"].append(False)  # 这里一定是False，finish=True已经被跳过
        
        # 传递额外字段
        if "extra_fields" in data:
            if "extra_fields" not in tool_groups[port]["data"]:
                tool_groups[port]["data"]["extra_fields"] = []
            tool_groups[port]["data"]["extra_fields"].append(data["extra_fields"][i])
        
        if "is_last_step" in data:
            if "is_last_step" not in tool_groups[port]["data"]:
                tool_groups[port]["data"]["is_last_step"] = []
            tool_groups[port]["data"]["is_last_step"].append(data["is_last_step"][i])
    
    # 统计finish跳过的数量
    finish_count = sum(1 for f in finish_flags if f)
    if finish_count > 0:
        print(f"⏭️ Skipped {finish_count} finished action(s)")
    
    # 调试：显示路由分组
    if tool_groups:
        print(f"🚀 Routing to {len(tool_groups)} different ports:")
        for port, group_info in tool_groups.items():
            print(f"  - Port {port}: {len(group_info['indices'])} requests")
    else:
        print(f"🚀 No tool routing needed (all actions finished)")
    
    # 并发调用不同端口
    if tool_groups:
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
                                raw_obs = resp_data["observations"][j]
                                
                                # 提取工具名称
                                action = actions[orig_idx]
                                tool_name = extract_tool_from_action(action)
                                
                                # 格式化工具输出
                                formatted_obs = format_tool_output(tool_name, port, raw_obs)
                                
                                # 如果格式化后的输出与原始输出不同，记录日志
                                if formatted_obs != raw_obs:
                                    print(f"  📝 Formatted {tool_name} output (port {port})")
                                    if isinstance(raw_obs, str) and len(raw_obs) > 200:
                                        print(f"     Original length: {len(raw_obs)} chars")
                                    if isinstance(formatted_obs, str):
                                        print(f"     Formatted preview: {formatted_obs[:150]}{'...' if len(formatted_obs) > 150 else ''}")
                                
                                results["observations"][orig_idx] = formatted_obs
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

# ==================== 健康检查 ====================

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

# ==================== 启动 ====================

if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("🚀 Starting Tool Router with Enhanced Output Formatting")
    print(f"📍 Listening on: http://0.0.0.0:5556")
    print(f"🔧 Tool Ports: {TOOL_PORTS}")
    print("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=5556, log_level="info")