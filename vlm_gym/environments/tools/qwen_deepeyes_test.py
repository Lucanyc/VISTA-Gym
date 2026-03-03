#!/usr/bin/env python3
"""
使用 Qwen2.5-VL 理解 ScienceQA 问题，自动决定需要查看的区域
"""

import os
import json
import re
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

import sys
sys.path.append('/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista')
from vlm_gym.environments.tools.deepeyes_tool import DeepEyesTool


def load_scienceqa_sample(json_path, sample_id="1"):
    """加载ScienceQA样本"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data[sample_id]


def extract_tool_call(response):
    """从响应中提取工具调用 - 支持多种格式"""
    
    # 方法1: 标准 JSON 代码块格式（优先）
    match = re.search(r'```(?:json)?\s*\n?(\[[^\]]+\])\s*\n?```', response, re.DOTALL)
    if match:
        try:
            json_str = match.group(1).strip()
            json_data = json.loads(json_str)
            if isinstance(json_data, list) and len(json_data) > 0:
                call = json_data[0]
                # 修正工具名
                if call.get("name") == "image_zoom_in":
                    call["name"] = "image_zoom_in_tool"
                print("✓ 找到标准 JSON 格式")
                return json.dumps(call)
        except Exception as e:
            print(f"解析 JSON 代码块失败: {e}")
    
    # 方法2: addCriterion 函数格式（Qwen 可能使用的格式）
    match = re.search(r'addCriterion\s*\(\s*"image_zoom_in"\s*,\s*(\{[^}]+\})\s*\)', response, re.DOTALL)
    if match:
        try:
            args_str = match.group(1)
            # 尝试解析参数
            args = json.loads(args_str)
            bbox = args.get("bbox_2d")
            
            if bbox and isinstance(bbox, list) and len(bbox) == 4:
                tool_call = {
                    "name": "image_zoom_in_tool",
                    "arguments": {"bbox_2d": bbox}
                }
                print("✓ 找到 addCriterion 格式")
                return json.dumps(tool_call)
        except Exception as e:
            print(f"解析 addCriterion 格式失败: {e}")
    
    # 方法3: 查找任何包含 bbox_2d 的 JSON 对象
    bbox_pattern = r'\{[^{}]*"bbox_2d"\s*:\s*\[[\d\s,]+\][^{}]*\}'
    matches = re.finditer(bbox_pattern, response)
    for match in matches:
        try:
            json_str = match.group(0)
            params = json.loads(json_str)
            if "bbox_2d" in params and isinstance(params["bbox_2d"], list):
                tool_call = {
                    "name": "image_zoom_in_tool",
                    "arguments": {"bbox_2d": params["bbox_2d"]}
                }
                print("✓ 找到独立 bbox JSON")
                return json.dumps(tool_call)
        except Exception as e:
            continue
    
    # 方法4: 直接查找坐标数组 [x1, y1, x2, y2]
    coord_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    match = re.search(coord_pattern, response)
    if match:
        coords = [int(match.group(i)) for i in range(1, 5)]
        # 验证坐标合理性
        if coords[2] > coords[0] and coords[3] > coords[1]:
            tool_call = {
                "name": "image_zoom_in_tool",
                "arguments": {"bbox_2d": coords}
            }
            print("✓ 找到裸坐标数组")
            return json.dumps(tool_call)
    
    # 调试信息
    print("✗ 未找到任何有效格式")
    print("响应中可能的坐标模式:")
    coords_in_response = re.findall(r'\[\d+,\s*\d+,\s*\d+,\s*\d+\]', response)
    for coord in coords_in_response[:3]:  # 只显示前3个
        print(f"  - {coord}")
    
    return None


def extract_answer(response, choices):
    """从响应中提取答案"""
    # 清理响应文本
    clean_response = response.strip()
    
    # 方法1: 完全匹配（整个响应就是答案）
    if clean_response in choices:
        return clean_response
    
    # 方法2: 在响应中查找选项（优先查找独立的词）
    for choice in choices:
        # 使用词边界确保完整匹配
        pattern = r'\b' + re.escape(choice) + r'\b'
        if re.search(pattern, response, re.IGNORECASE):
            return choice
    
    # 方法3: 查找 "answer is X" 或 "X is the answer" 模式
    for choice in choices:
        patterns = [
            rf'answer\s+is\s+{re.escape(choice)}',
            rf'{re.escape(choice)}\s+is\s+the\s+answer',
            rf'answer:\s*{re.escape(choice)}',
            rf'{re.escape(choice)}\s+is\s+farthest\s+north'
        ]
        for pattern in patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return choice
    
    return None


def solve_scienceqa_with_qwen_deepeyes(sample_id="1"):
    """
    使用 Qwen2.5-VL + DeepEyes 解决 ScienceQA 问题
    """
    
    # 1. 加载数据集
    dataset_base = "/data/wang/meng/GYM-Work/dataset/ScienceQA"
    json_path = os.path.join(dataset_base, "reformatted/train.json")
    sample = load_scienceqa_sample(json_path, sample_id)
    
    print("=== ScienceQA 样本信息 ===")
    print(f"问题: {sample['question']}")
    print(f"选项: {sample['choices']}")
    print(f"正确答案: {sample['answer']} (不会告诉模型)")
    
    # 2. 加载图像
    image_path = os.path.join(dataset_base, sample['image'])
    if not os.path.exists(image_path):
        print(f"错误：图像不存在 {image_path}")
        return None
        
    image = Image.open(image_path)
    width, height = image.size
    print(f"\n图像路径: {image_path}")
    print(f"图像尺寸: {width}x{height}")
    
    # 3. 加载 Qwen2.5-VL 模型
    print("\n加载 Qwen2.5-VL 模型...")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    model = AutoModelForImageTextToText.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 4. 初始化 DeepEyes
    tool = DeepEyesTool()
    tool.reset(image)
    
    # 5. 第一轮：让 Qwen2.5-VL 分析问题并决定需要查看的区域
    print("\n=== 第一轮：Qwen2.5-VL 分析问题 ===")
    
    # 构建消息 - 强约束格式
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"""You need to answer: {sample['question']}
Choices: {', '.join(sample['choices'])}

The image size is {width}x{height}.

You have access to a zoom tool. To zoom into a region, you MUST return your tool call in the following format:

```json
[{{"name": "image_zoom_in", "arguments": {{"bbox_2d": [left, top, right, bottom]}}}}]
```

IMPORTANT:
- You MUST use a JSON code block exactly like above
- Do NOT use function-style syntax like addCriterion(...)
- The bbox coordinates must be within the image bounds: [0, 0, {width}, {height}]
- Just output the tool call in the JSON format, nothing else

Example for zooming to the northern third of the image:
```json
[{{"name": "image_zoom_in", "arguments": {{"bbox_2d": [0, 0, {width}, {height//3}]}}}}]
```

Now, based on the question about which state is farthest north, what region should we zoom into?"""}
            ]
        }
    ]
    
    # 处理输入
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt"
    ).to(model.device)
    
    # 生成响应
    print("\n生成模型响应中...")
    outputs = model.generate(**inputs, max_new_tokens=512)
    response = processor.decode(outputs[0], skip_special_tokens=True)
    
    # 提取模型的响应部分
    if "assistant:" in response:
        response = response.split("assistant:")[-1].strip()
    elif "assistant\n" in response:
        response = response.split("assistant\n")[-1].strip()
    
    print(f"\nQwen2.5-VL 响应:")
    print(response[:500] + "..." if len(response) > 500 else response)
    
    # 检查是否包含特殊格式
    if "addCriterion" in response:
        print("\n注意：模型使用了 addCriterion 格式")
    
    # 6. 提取工具调用
    tool_call = extract_tool_call(response)
    
    if tool_call:
        print(f"\n提取到工具调用: {tool_call}")
        
        # 7. 执行 DeepEyes 缩放
        print("\n执行 DeepEyes 缩放...")
        action = f"<tool_call>{tool_call}</tool_call>"
        result = tool.execute(action)
        
        if result.get('success'):
            print("✓ 缩放成功!")
            print(f"  使用坐标: {result.get('bbox_used')}")
            print(f"  新尺寸: {result.get('new_size')}")
            
            # 保存缩放结果
            zoomed_image = result.get('image')
            if zoomed_image:
                zoom_path = f"scienceqa_{sample_id}_zoom.png"
                zoomed_image.save(zoom_path)
                print(f"  缩放图像已保存到: {zoom_path}")
                
                # 8. 第二轮：基于缩放后的图像进行分析
                print("\n=== 第二轮：分析缩放后的图像 ===")
                
                messages_round2 = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": zoomed_image},
                            {"type": "text", "text": f"""This is the zoomed image showing the northern part of the map.
Question: {sample['question']}
Choices: {', '.join(sample['choices'])}

Based on what you can see in this zoomed image, which state is positioned farthest north?
Please respond with ONLY the state name from the given choices. Nothing else."""}
                        ]
                    }
                ]
                
                # 第二轮推理
                text2 = processor.apply_chat_template(messages_round2, tokenize=False, add_generation_prompt=True)
                inputs2 = processor(
                    text=text2,
                    images=zoomed_image,
                    return_tensors="pt"
                ).to(model.device)
                
                outputs2 = model.generate(**inputs2, max_new_tokens=256)
                response2 = processor.decode(outputs2[0], skip_special_tokens=True)
                
                if "assistant:" in response2:
                    response2 = response2.split("assistant:")[-1].strip()
                elif "assistant\n" in response2:
                    response2 = response2.split("assistant\n")[-1].strip()
                
                print(f"\n第二轮响应:\n{response2}")
                
                # 提取答案
                answer = extract_answer(response2, sample['choices'])
                if answer:
                    print(f"\n预测答案: {answer}")
                    print(f"正确答案: {sample['answer']}")
                    print(f"结果: {'✓ 正确' if answer == sample['answer'] else '✗ 错误'}")
                    
                    return {
                        'sample_id': sample_id,
                        'question': sample['question'],
                        'predicted': answer,
                        'correct': sample['answer'],
                        'success': answer == sample['answer']
                    }
                else:
                    print("\n✗ 模型未能给出明确答案")
        else:
            print(f"✗ DeepEyes 执行失败: {result.get('error')}")
            print(f"  错误类型: {result.get('error_type')}")
            if 'original_bbox' in result:
                print(f"  原始边界框: {result.get('original_bbox')}")
    else:
        print("\n✗ 模型未生成有效的工具调用")
        print("请检查模型输出格式是否正确")
    
    return None


def test_multiple_samples():
    """测试多个样本"""
    sample_ids = ["1", "2", "3"]
    results = []
    
    for sample_id in sample_ids:
        print(f"\n{'='*70}")
        print(f"测试样本 {sample_id}")
        print(f"{'='*70}")
        
        try:
            result = solve_scienceqa_with_qwen_deepeyes(sample_id)
            if result:
                results.append(result)
        except Exception as e:
            print(f"处理样本 {sample_id} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 统计结果
    if results:
        correct = sum(1 for r in results if r['success'])
        print(f"\n\n{'='*70}")
        print("测试结果汇总")
        print(f"{'='*70}")
        print(f"总样本数: {len(results)}")
        print(f"正确数: {correct}")
        print(f"准确率: {correct/len(results)*100:.1f}%")
        
        print("\n详细结果:")
        for r in results:
            status = "✓" if r['success'] else "✗"
            print(f"{status} 样本 {r['sample_id']}: {r['predicted']} (正确: {r['correct']})")


if __name__ == "__main__":
    # 测试单个样本
    result = solve_scienceqa_with_qwen_deepeyes("1")
    
    # 或测试多个样本
    # test_multiple_samples()
    
    if result:
        print(f"\n测试完成!")
    else:
        print(f"\n测试未能完成")