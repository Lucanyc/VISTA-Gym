#!/usr/bin/env python3
import requests
import json
import re

def test_tool_call():
    # API配置
    API_URL = "http://localhost:5000/chat/completions"
    MODEL_NAME = "/home/menglu/Online-GRPO/verl-tool/Evaluation/trained-qwen2.5vl-chartqa-online-grpo-new"
    
    # 测试图像
    test_image = "/home/menglu/Online-GRPO/verl-tool/data/chartqa_dataset/chartqa/test/png/166.png"
    
    # 使用正确的工具调用格式
    test_prompts = [
        # 测试1：analyze任务
        'Please analyze this chart. Use <tool_call>{"tool": "chartmoe", "task": "analyze"}</tool_call>',
        
        # 测试2：to_table任务
        'Convert this chart to a table. Use <tool_call>{"tool": "chartmoe", "task": "to_table"}</tool_call>',
        
        # 测试3：extract_data任务
        'Extract data from this chart. Use <tool_call>{"tool": "chartmoe", "task": "extract_data"}</tool_call>',
        
        # 测试4：让模型自己选择
        'What is the highest value in this chart? Please use the chartmoe tool to analyze it.',
        
        # 测试5：更自然的提示
        'I need to understand this chart. Can you help me analyze it using the appropriate tool?'
    ]
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n{'='*70}")
        print(f"Test {i+1}")
        print(f"Prompt: {prompt}")
        print('='*70)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"file://{test_image}"}},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 1000  # 增加以看到完整的工具交互
        }
        
        try:
            response = requests.post(API_URL, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            full_response = result['choices'][0]['message']['content']
            
            print("\nFull Response:")
            print("-" * 70)
            print(full_response)
            print("-" * 70)
            
            # 修复后的检测逻辑 - 只检查 <tool_call> 开始标签
            if "<tool_call>" in full_response:
                print("\n✅ Tool call detected!")
                
                # 使用正则表达式匹配 <tool_call>...stop 或 <tool_call>...</tool_call>
                tool_calls = re.findall(r'<tool_call>(.*?)(?:stop|</tool_call>)', full_response, re.DOTALL)
                
                for idx, call in enumerate(tool_calls):
                    print(f"\nTool call {idx+1}: {call.strip()}")
                    try:
                        # 尝试解析JSON
                        tool_data = json.loads(call.strip())
                        print(f"  - Tool: {tool_data.get('tool')}")
                        print(f"  - Task: {tool_data.get('task')}")
                    except Exception as e:
                        print(f"  - JSON parsing error: {e}")
                
                # 检查工具输出
                if "output" in full_response:
                    print("\n✅ Tool output detected!")
                    # 提取输出内容 - 匹配 output...stop 或到字符串结尾
                    output_pattern = r'output\s*(.*?)(?:stop|$)'
                    outputs = re.findall(output_pattern, full_response, re.DOTALL)
                    for idx, output in enumerate(outputs):
                        output_text = output.strip()
                        print(f"\nTool output {idx+1}:")
                        # 如果输出太长，只显示前200个字符
                        if len(output_text) > 200:
                            print(output_text[:200] + "...")
                        else:
                            print(output_text)
                        
                        # 检查是否包含表格格式
                        if "|---|" in output_text or "| Entity |" in output_text:
                            print("  ✓ Table format detected")
                            
            else:
                print("\n❌ No tool call detected")
                # 额外检查是否有工具输出的痕迹
                if "output" in full_response or "|---|" in full_response:
                    print("⚠️  But tool output format detected - might be implicit tool usage")
                
        except Exception as e:
            print(f"\nError: {e}")

def test_manual_tool_call():
    """直接测试工具服务器"""
    print("\n" + "="*70)
    print("Testing Tool Server Directly")
    print("="*70)
    
    # 测试不同的任务
    tasks = [
        {"tool": "chartmoe", "task": "analyze"},
        {"tool": "chartmoe", "task": "to_table"},
        {"tool": "chartmoe", "task": "extract_data"}
    ]
    
    for task in tasks:
        print(f"\nTesting task: {task}")
        
        payload = {
            "trajectory_ids": ["test-123"],
            "actions": [f'<tool_call>{json.dumps(task)}</tool_call>'],
            "finish": [False],
            "image_path": [[{"image_url": "/home/menglu/Online-GRPO/verl-tool/data/chartqa_dataset/chartqa/test/png/166.png"}]]
        }
        
        try:
            response = requests.post(
                "http://localhost:5556/get_observation",
                json=payload,
                timeout=30
            )
            print(f"Response: {response.json()}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    print("Testing tool call functionality with correct format...")
    test_tool_call()
    test_manual_tool_call()