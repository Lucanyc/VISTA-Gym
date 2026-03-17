import requests
import json

# 测试 API 服务
base_url = "http://localhost:5000"  # 根据您的实际端口调整
MODEL_NAME = "/home/menglu/Online-GRPO/verl-tool/Evaluation/trained-qwen2.5vl-chartqa-online-grpo-new"  

# 测试1: 带图像的聊天请求
print("=== 测试1: 带图像的聊天请求 ===")
image_path = "/home/menglu/Online-GRPO/verl-tool/data/chartqa_dataset/chartqa/test/png/166.png"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"file://{image_path}"}},
            {"type": "text", "text": "What is the highest value in this chart? Use the chartmoe tool to analyze it."}
        ]
    }
]

payload = {
    "model": MODEL_NAME, 
    "messages": messages,
    "temperature": 0.7,
    "max_tokens": 200
}

try:
    response = requests.post(f"{base_url}/chat/completions", json=payload)
    response.raise_for_status()
    result = response.json()
    print(json.dumps(result, indent=2))
except Exception as e:
    print(f"错误: {e}")
    if hasattr(e, 'response'):
        print(f"响应内容: {e.response.text}")

# 测试2: 纯文本请求（确保向后兼容）
print("\n=== 测试2: 纯文本请求 ===")
messages = [
    {
        "role": "user",
        "content": "What is 2+2?"
    }
]

payload = {
    "model": MODEL_NAME,
    "messages": messages,
    "temperature": 0.0,
    "max_tokens": 50
}

try:
    response = requests.post(f"{base_url}/chat/completions", json=payload)
    response.raise_for_status()
    result = response.json()
    print(json.dumps(result, indent=2))
except Exception as e:
    print(f"错误: {e}")
    if hasattr(e, 'response'):
        print(f"响应内容: {e.response.text}")