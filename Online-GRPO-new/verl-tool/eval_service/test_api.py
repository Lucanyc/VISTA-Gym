#!/usr/bin/env python
"""
test_api_format.py - 测试API接受的消息格式
"""

import pandas as pd
import json
import numpy as np
from openai import OpenAI

# 配置
MODEL_PATH = "/path/to/your/global_step_200/actor/huggingface"  # 修改为你的模型路径
API_URL = "http://localhost:5000/v1"
TEST_PARQUET = "/mnt/nfs/meng/Online-GRPO-new/verl-tool/eval_service/test_dataset_transfer/processed_official_test_format/test_dataset.parquet"

# 初始化客户端
client = OpenAI(base_url=API_URL, api_key="not-needed")

def convert_numpy(obj):
    """递归转换numpy类型为Python原生类型"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy(val) for key, val in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    else:
        return obj

# 读取一个样本
df = pd.read_parquet(TEST_PARQUET)
sample_raw = df.iloc[0].to_dict()

# 转换numpy类型
sample = convert_numpy(sample_raw)

print("样本信息:")
print(f"  data_source: {sample['data_source']}")
print(f"  images: {sample['images']}")
print(f"  ground_truth: {sample['reward_model']['ground_truth']}")
print("\n查看prompt结构:")
for i, msg in enumerate(sample['prompt']):
    print(f"  消息{i}: role={msg['role']}, content长度={len(str(msg['content']))}")
print("\n" + "="*50 + "\n")

# 格式1: 直接使用<image>标记
def test_format1():
    print("测试格式1: 直接使用<image>标记")
    messages = sample['prompt']  # 直接使用原始prompt
    
    print("发送的消息（前100字符）:")
    for msg in messages:
        print(f"  {msg['role']}: {str(msg['content'])[:100]}...")
    
    try:
        response = client.chat.completions.create(
            model=MODEL_PATH,
            messages=messages,
            temperature=0.7,
            max_tokens=2048
        )
        print("\n✓ 格式1成功!")
        print(f"响应: {response.choices[0].message.content[:200]}...")
        return True
    except Exception as e:
        print(f"\n✗ 格式1失败: {e}")
        return False

# 格式2: 转换为多模态格式
def test_format2():
    print("\n测试格式2: 多模态格式")
    messages = []
    
    for msg in sample['prompt']:
        if msg['role'] == 'system':
            messages.append(msg)
        elif msg['role'] == 'user':
            # 解析content中的<image>和文本
            content = msg['content']
            if '<image>' in content:
                # 分离图片标记和文本
                text = content.replace('<image>', '').strip()
                # 构建多模态content
                formatted_content = [
                    {"type": "image", "image": sample['images'][0]['image']},
                    {"type": "text", "text": text}
                ]
                messages.append({
                    "role": "user",
                    "content": formatted_content
                })
            else:
                messages.append(msg)
    
    print("发送的消息结构:")
    for msg in messages:
        if msg['role'] == 'system':
            print(f"  system: {msg['content'][:100]}...")
        else:
            print(f"  user: {len(msg['content'])} items")
            for item in msg['content']:
                if item['type'] == 'text':
                    print(f"    - text: {item['text'][:100]}...")
                else:
                    print(f"    - image: {item['image']}")
    
    try:
        response = client.chat.completions.create(
            model=MODEL_PATH,
            messages=messages,
            temperature=0.7,
            max_tokens=2048
        )
        print("\n✓ 格式2成功!")
        print(f"响应: {response.choices[0].message.content[:200]}...")
        return True
    except Exception as e:
        print(f"\n✗ 格式2失败: {e}")
        return False

# 执行测试
if __name__ == "__main__":
    print("开始测试API格式...\n")
    
    # 测试两种格式
    format1_ok = test_format1()
    print("\n" + "-"*50)
    format2_ok = test_format2()
    
    print("\n" + "="*50)
    print("测试结果:")
    if format1_ok and not format2_ok:
        print("→ 使用格式1 (直接<image>标记)")
    elif format2_ok and not format1_ok:
        print("→ 使用格式2 (多模态格式)")
    elif format1_ok and format2_ok:
        print("→ 两种格式都支持")
    else:
        print("→ 两种格式都失败，需要检查API服务")
        print("\n可能的问题:")
        print("1. API服务未启动")
        print("2. 模型路径不正确")
        print("3. 图片路径无法访问")