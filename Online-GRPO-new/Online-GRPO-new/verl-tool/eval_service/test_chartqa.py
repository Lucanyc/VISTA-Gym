import requests
import json
import os
from typing import List, Dict

# API配置
BASE_URL = "http://localhost:5000"
MODEL_NAME = "/home/menglu/Online-GRPO/verl-tool/Evaluation/trained-qwen2.5vl-chartqa-online-grpo-new"

def test_single_chartqa(image_path: str, question: str, ground_truth: str = None) -> Dict:
    """
    测试单个ChartQA问题
    
    Args:
        image_path: 图表图片路径
        question: 关于图表的问题
        ground_truth: 正确答案（可选，用于对比）
    
    Returns:
        包含预测结果的字典
    """
    # 确保图片文件存在
    if not os.path.exists(image_path):
        return {"error": f"Image file not found: {image_path}"}
    
    # 构建请求
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"file://{image_path}"}},
                {"type": "text", "text": question}
            ]
        }
    ]
    
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.1,  # 降低温度以获得更确定的答案
        "max_tokens": 500
    }
    
    try:
        response = requests.post(f"{BASE_URL}/chat/completions", json=payload)
        response.raise_for_status()
        result = response.json()
        
        # 提取模型的回答
        model_answer = result["choices"][0]["message"]["content"]
        
        return {
            "question": question,
            "model_answer": model_answer,
            "ground_truth": ground_truth,
            "image_path": image_path,
            "full_response": result
        }
        
    except requests.exceptions.HTTPError as e:
        return {
            "error": f"HTTP Error: {e}",
            "response": e.response.text if hasattr(e, 'response') else None
        }
    except Exception as e:
        return {"error": f"Error: {str(e)}"}

def load_chartqa_data(json_file: str) -> List[Dict]:
    """加载ChartQA测试数据"""
    with open(json_file, 'r') as f:
        return json.load(f)

def main():
    print("=== ChartQA 测试 ===\n")
    
    # 测试几个示例
    test_cases = [
        {
            "image_path": "/home/menglu/Online-GRPO/verl-tool/data/chartqa_dataset/chartqa/test/png/41699051005347.png",
            "question": "How many food item is shown in the bar graph?",
            "answer": "14"
        },
        {
            "image_path": "/home/menglu/Online-GRPO/verl-tool/data/chartqa_dataset/chartqa/test/png/41699051005347.png",
            "question": "What is the difference in value between Lamb and Corn?",
            "answer": "0.57"
        },
        {
            "image_path": "/home/menglu/Online-GRPO/verl-tool/data/chartqa_dataset/chartqa/test/png/41810321001157.png",
            "question": "How many bars are shown in the chart?",
            "answer": "3"
        }
    ]
    
    # 测试每个案例
    for i, test_case in enumerate(test_cases):
        print(f"--- 测试案例 {i+1} ---")
        print(f"图片: {os.path.basename(test_case['image_path'])}")
        print(f"问题: {test_case['question']}")
        print(f"标准答案: {test_case['answer']}")
        
        result = test_single_chartqa(
            test_case['image_path'],
            test_case['question'],
            test_case['answer']
        )
        
        if "error" in result:
            print(f"错误: {result['error']}")
            if "response" in result:
                print(f"响应详情: {result['response']}")
        else:
            print(f"模型回答: {result['model_answer']}")
            
            # 检查是否使用了工具
            if "</tool_call>" in result['model_answer'] or "<tool_call>" in result['model_answer']:
                print("✓ 模型使用了工具调用")
            else:
                print("✗ 模型未使用工具调用")
        
        print("\n")

if __name__ == "__main__":
    main()
