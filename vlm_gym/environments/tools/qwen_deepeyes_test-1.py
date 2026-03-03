#!/usr/bin/env python3
"""
改进版本 - 修复响应解码和生成参数问题
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


def extract_bbox_from_response(response):
    """从响应中提取边界框坐标"""
    # 多种模式匹配
    patterns = [
        r'<box>\s*\[([0-9]+),\s*([0-9]+),\s*([0-9]+),\s*([0-9]+)\]\s*</box>',
        r'<box>\s*\[([0-9,\s]+)\]\s*</box>',
        r'\[([0-9]+),\s*([0-9]+),\s*([0-9]+),\s*([0-9]+)\]',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            if len(match.groups()) == 4:
                # 直接获取四个坐标
                return [int(match.group(i)) for i in range(1, 5)]
            elif len(match.groups()) == 1:
                # 需要分割字符串
                coords_str = match.group(1)
                coords = [int(x.strip()) for x in coords_str.split(',')]
                if len(coords) == 4:
                    return coords
    return None


def test_qwen_with_better_decoding(sample_id="1"):
    """使用改进的解码方法测试 Qwen2.5-VL"""
    
    # 加载数据
    dataset_base = "/data/wang/meng/GYM-Work/dataset/ScienceQA"
    json_path = os.path.join(dataset_base, "reformatted/train.json")
    sample = load_scienceqa_sample(json_path, sample_id)
    
    # 加载图像
    image_path = os.path.join(dataset_base, sample['image'])
    image = Image.open(image_path)
    width, height = image.size
    
    print(f"问题: {sample['question']}")
    print(f"选项: {sample['choices']}")
    print(f"图像尺寸: {width}x{height}\n")
    
    # 加载模型
    print("加载 Qwen2.5-VL 模型...")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    model = AutoModelForImageTextToText.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 使用不同的提示策略
    test_configs = [
        {
            "name": "直接坐标",
            "prompt": f"Output: <box>[0,0,{width},{height//3}]</box>",
            "temperature": 0.01,
            "do_sample": False
        },
        {
            "name": "引导式",
            "prompt": f"""To find the northernmost state, I need to examine the top portion of the map.
<box>[0,0,{width},{height//3}]</box>""",
            "temperature": 0.1,
            "do_sample": False
        },
        {
            "name": "问答式",
            "prompt": f"""Question: {sample['question']}
To answer this, I'll zoom to the northern region.
Coordinates: <box>[0,0,{width},{height//3}]</box>""",
            "temperature": 0.1,
            "do_sample": False
        },
    ]
    
    for config in test_configs:
        print(f"\n{'='*50}")
        print(f"测试: {config['name']}")
        print(f"温度: {config['temperature']}")
        print(f"{'='*50}")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": config['prompt']}
                ]
            }
        ]
        
        # 应用聊天模板
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 记录输入长度，用于后续只解码新生成的部分
        inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)
        input_length = inputs['input_ids'].shape[-1]
        
        # 生成响应 - 使用更保守的参数
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=100,  # 减少到100
                temperature=config['temperature'],
                do_sample=config['do_sample'],
                top_p=0.9,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        
        # 只解码新生成的部分
        generated_ids = outputs[0][input_length:]
        response = processor.decode(generated_ids, skip_special_tokens=True)
        
        print(f"生成的响应:\n{response}\n")
        
        # 提取坐标
        bbox = extract_bbox_from_response(response)
        if bbox:
            print(f"✓ 成功提取坐标: {bbox}")
        else:
            print("✗ 未能提取坐标")


def test_integrated_with_deepeyes(sample_id="1"):
    """完整的 DeepEyes 集成测试"""
    
    # 加载数据
    dataset_base = "/data/wang/meng/GYM-Work/dataset/ScienceQA"
    json_path = os.path.join(dataset_base, "reformatted/train.json")
    sample = load_scienceqa_sample(json_path, sample_id)
    
    # 加载图像
    image_path = os.path.join(dataset_base, sample['image'])
    image = Image.open(image_path)
    width, height = image.size
    
    print(f"\n{'='*60}")
    print(f"完整测试: {sample['question']}")
    print(f"选项: {sample['choices']}")
    print(f"正确答案: {sample['answer']}")
    print(f"{'='*60}\n")
    
    # 初始化 DeepEyes
    tool = DeepEyesTool()
    tool.reset(image)
    
    # 加载模型
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    model = AutoModelForImageTextToText.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 第一步：获取要放大的区域
    print("第一步：识别需要放大的区域...")
    
    # 使用简单直接的提示
    first_prompt = f"<box>[0,0,{width},{height//3}]</box>"
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": first_prompt}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)
    input_length = inputs['input_ids'].shape[-1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.01,
            do_sample=False,
        )
    
    generated_ids = outputs[0][input_length:]
    response = processor.decode(generated_ids, skip_special_tokens=True)
    
    print(f"模型响应: {response}")
    
    # 提取坐标或使用默认值
    bbox = extract_bbox_from_response(first_prompt + " " + response)
    if not bbox:
        bbox = [0, 0, width, height//3]
        print(f"使用默认坐标: {bbox}")
    else:
        print(f"提取到坐标: {bbox}")
    
    # 执行放大
    tool_call = json.dumps({
        "name": "image_zoom_in_tool",
        "arguments": {"bbox_2d": bbox}
    })
    action = f"<tool_call>{tool_call}</tool_call>"
    result = tool.execute(action)
    
    if result.get('success'):
        zoomed_image = result.get('image')
        print(f"放大成功，新尺寸: {result.get('new_size')}")
        
        # 保存放大的图像以供检查
        zoomed_image.save(f"zoomed_region_sample_{sample_id}.png")
        
        # 第二步：在放大的图像上回答问题
        print("\n第二步：分析放大的图像...")
        
        # 使用简单明确的提示
        second_prompt = f"From the options {', '.join(sample['choices'])}, which state is shown? Answer with just the state name:"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": zoomed_image},
                    {"type": "text", "text": second_prompt}
                ]
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=zoomed_image, return_tensors="pt").to(model.device)
        input_length = inputs['input_ids'].shape[-1]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.1,
                do_sample=False,
            )
        
        generated_ids = outputs[0][input_length:]
        final_answer = processor.decode(generated_ids, skip_special_tokens=True).strip()
        
        print(f"\n最终答案: {final_answer}")
        print(f"正确答案: {sample['answer']}")
        
        # 检查答案是否在选项中
        if final_answer in sample['choices']:
            print(f"✓ 答案在选项中")
            if final_answer == sample['answer']:
                print(f"✓ 答案正确！")
            else:
                print(f"✗ 答案错误")
        else:
            print(f"✗ 答案不在选项中")
            
    else:
        print("放大失败")


def debug_tokenizer_behavior():
    """调试 tokenizer 的行为"""
    print("\n调试 tokenizer 行为...")
    print("="*60)
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    
    # 创建测试消息
    test_image = Image.new('RGB', (100, 100), color='blue')
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": test_image},
                {"type": "text", "text": "Test message"}
            ]
        }
    ]
    
    # 应用聊天模板
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"聊天模板输出:\n{text}\n")
    
    # tokenize
    inputs = processor(text=text, images=test_image, return_tensors="pt")
    print(f"输入张量形状: {inputs['input_ids'].shape}")
    
    # 解码查看内容
    decoded_input = processor.decode(inputs['input_ids'][0])
    print(f"解码的输入:\n{decoded_input}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "improved":
            # 测试改进的解码方法
            test_qwen_with_better_decoding()
        elif sys.argv[1] == "integrated":
            # 完整的集成测试
            test_integrated_with_deepeyes()
        elif sys.argv[1] == "debug":
            # 调试 tokenizer
            debug_tokenizer_behavior()
    else:
        # 默认运行改进的测试
        print("1. 测试改进的解码方法...")
        test_qwen_with_better_decoding()
        
        print("\n\n2. 运行完整集成测试...")
        test_integrated_with_deepeyes()