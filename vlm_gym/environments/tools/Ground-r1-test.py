import re
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image, ImageDraw
import torch
import os

# 定义输出目录
OUTPUT_DIR = "/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/vlm_gym/environments/tools/crop_image/"

def test_qwen25_simple():
    """基于之前成功的简单方法"""
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📁 Output directory: {OUTPUT_DIR}")
    
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    image_path = "/data/wang/meng/GYM-Work/dataset/ScienceQA/images/train/train_000000.png"
    image = Image.open(image_path)
    
    # 使用之前成功的prompt
    prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Question: Which of these states is farthest north?
Options: A. West Virginia, B. Louisiana, C. Arizona, D. Oklahoma

Output format:
<think>reasoning</think>
<box>[x1,y1,x2,y2]</box>
<answer>answer</answer>

Begin:<|im_end|>
<|im_start|>assistant
<think>"""
    
    print("🔥 Testing Qwen2.5-VL with simple approach...")
    print("="*60)
    
    # 处理输入
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.2,
            do_sample=True
        )
    
    response = processor.decode(generated_ids[0], skip_special_tokens=True)
    
    # 提取生成的部分
    if "<think>" in response:
        response = response.split("<think>", 1)[1]
        response = "<think>" + response
    
    print("Response:")
    print(response)
    print("="*60)
    
    # 解析响应（修复bbox格式）
    def parse_and_fix_response(text):
        # 先分离出实际生成的内容（在assistant标记之后的内容）
        if "assistant" in text:
            # 找到最后一个assistant标记后的内容
            parts = text.split("assistant")
            if len(parts) > 1:
                text = parts[-1]
        
        # 修复常见的bbox格式问题
        # <box[648,250,733,269]></box> -> <box>[648,250,733,269]</box>
        text = re.sub(r'<box\[(\d+),(\d+),(\d+),(\d+)\]></box>', r'<box>[\1,\2,\3,\4]</box>', text)
        
        # 提取组件 - 使用findall获取所有匹配，然后取最后一个（实际生成的）
        results = {}
        
        # 提取thinking - 获取所有think标签，取最后一个
        think_pattern = r'<think>(.*?)</think>'
        think_matches = re.findall(think_pattern, text, re.DOTALL)
        if think_matches:
            # 过滤掉只包含"reasoning"的示例
            real_thinks = [t.strip() for t in think_matches if t.strip() != "reasoning"]
            results['thinking'] = real_thinks[-1] if real_thinks else None
        else:
            results['thinking'] = None
        
        # 提取bbox
        bbox_pattern = r'<box>\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*</box>'
        bbox_match = re.search(bbox_pattern, text)
        results['bbox'] = [int(x) for x in bbox_match.groups()] if bbox_match else None
        
        # 提取answer - 获取所有answer标签，取最后一个
        answer_pattern = r'<answer>(.*?)</answer>'
        answer_matches = re.findall(answer_pattern, text, re.DOTALL)
        if answer_matches:
            # 过滤掉只包含"answer"的示例
            real_answers = [a.strip() for a in answer_matches if a.strip() != "answer"]
            results['answer'] = real_answers[-1] if real_answers else None
        else:
            results['answer'] = None
        
        return results
    
    parsed = parse_and_fix_response(response)
    
    print("\n📊 Parsed Results:")
    if parsed['thinking']:
        print(f"Thinking: {parsed['thinking'][:100]}...")
    else:
        print("Thinking: None")
    print(f"BBox: {parsed['bbox']}")
    print(f"Answer: {parsed['answer']}")
    
    # 可视化bbox
    if parsed['bbox']:
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        draw.rectangle(parsed['bbox'], outline='red', width=3)
        
        # 保存到指定目录
        save_path = os.path.join(OUTPUT_DIR, "qwen25_bbox_simple.png")
        img_copy.save(save_path)
        print(f"\n✅ Saved visualization to {save_path}")
        
        # 显示裁剪区域
        x1, y1, x2, y2 = parsed['bbox']
        cropped = image.crop((x1, y1, x2, y2))
        
        # 保存裁剪图像到指定目录
        cropped_path = os.path.join(OUTPUT_DIR, "qwen25_cropped_simple.png")
        cropped.save(cropped_path)
        print(f"📦 Cropped region saved to {cropped_path}")
        
        # 打印图像尺寸信息
        print(f"📐 Original image size: {image.size}")
        print(f"📐 Cropped region size: {cropped.size}")
    
    return parsed

# 简单的多轮实现
class SimpleGroundR1:
    def __init__(self):
        self.model = AutoModelForVision2Seq.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            trust_remote_code=True
        )
    
    def fix_bbox_format(self, text):
        """修复bbox格式"""
        return re.sub(r'<box\[(\d+),(\d+),(\d+),(\d+)\]></box>', r'<box>[\1,\2,\3,\4]</box>', text)
    
    def extract_bbox(self, text):
        """提取bbox坐标"""
        # 分离实际生成的内容
        if "assistant" in text:
            parts = text.split("assistant")
            if len(parts) > 1:
                text = parts[-1]
        
        text = self.fix_bbox_format(text)
        pattern = r'<box>\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*</box>'
        match = re.search(pattern, text)
        return [int(x) for x in match.groups()] if match else None
    
    def extract_answer(self, text):
        """提取答案"""
        # 分离实际生成的内容
        if "assistant" in text:
            parts = text.split("assistant")
            if len(parts) > 1:
                text = parts[-1]
        
        pattern = r'<answer>(.*?)</answer>'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            # 过滤掉示例中的"answer"
            real_answers = [a.strip() for a in matches if a.strip() != "answer"]
            return real_answers[-1] if real_answers else None
        return None
    
    def run_inference(self, image, prompt):
        """运行推理"""
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.2,
                do_sample=True
            )
        
        response = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        
        # 更准确地提取生成的部分
        # 找到最后一个assistant标记后的内容
        if "assistant" in response:
            parts = response.split("assistant")
            if len(parts) > 1:
                response = parts[-1].strip()
        
        return response
    
    def multi_round_grounding(self, image_path, question, choices=None, max_rounds=3):
        """多轮grounding"""
        original_image = Image.open(image_path)
        current_image = original_image
        
        all_bboxes = []
        all_responses = []
        
        for round_idx in range(max_rounds):
            print(f"\n📍 Round {round_idx + 1}:")
            
            # 构造prompt
            if round_idx == 0:
                # 第一轮
                choice_text = f"Options: {', '.join([f'{chr(65+i)}. {c}' for i, c in enumerate(choices)])}" if choices else ""
                prompt = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Question: {question}
{choice_text}

Output format:
<think>reasoning</think>
<box>[x1,y1,x2,y2]</box>
<answer>answer</answer>

Begin:<|im_end|>
<|im_start|>assistant
<think>"""
            else:
                # 后续轮次
                prompt = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Based on the zoomed region, continue answering: {question}

Provide your final answer in <answer> tags.<|im_end|>
<|im_start|>assistant
"""
            
            # 运行推理
            response = self.run_inference(current_image, prompt)
            all_responses.append(response)
            
            print(f"Response: {response[:200]}...")
            
            # 提取答案
            answer = self.extract_answer(response)
            if answer:
                print(f"✅ Final Answer: {answer}")
                return {
                    'answer': answer,
                    'responses': all_responses,
                    'bboxes': all_bboxes,
                    'rounds': round_idx + 1
                }
            
            # 提取bbox
            bbox = self.extract_bbox(response)
            if bbox:
                print(f"📦 Found bbox: {bbox}")
                all_bboxes.append(bbox)
                
                # 裁剪图像用于下一轮
                x1, y1, x2, y2 = bbox
                current_image = original_image.crop((x1, y1, x2, y2))
                
                # 保存多轮的裁剪图像到指定目录
                cropped_path = os.path.join(OUTPUT_DIR, f"round_{round_idx+1}_cropped.png")
                current_image.save(cropped_path)
                print(f"📸 Saved round {round_idx+1} cropped image to {cropped_path}")
                
                # 保存带边界框的原图
                img_with_box = original_image.copy()
                draw = ImageDraw.Draw(img_with_box)
                draw.rectangle(bbox, outline='red', width=3)
                bbox_path = os.path.join(OUTPUT_DIR, f"round_{round_idx+1}_bbox.png")
                img_with_box.save(bbox_path)
                print(f"📸 Saved round {round_idx+1} bbox visualization to {bbox_path}")
            else:
                print("⚠️ No bbox found")
                break
        
        return {
            'answer': None,
            'responses': all_responses,
            'bboxes': all_bboxes,
            'rounds': len(all_responses)
        }

if __name__ == "__main__":
    # 测试简单版本
    result = test_qwen25_simple()
    
    if result['answer'] and result['answer'] != "answer":  # 确保不是示例文本
        print(f"\n🎉 Success! The model correctly identified: {result['answer']}")
        
        # 如果成功，可以测试多轮版本
        print("\n" + "="*60)
        print("Testing multi-round grounding...")
        
        grounding = SimpleGroundR1()
        multi_result = grounding.multi_round_grounding(
            image_path="/data/wang/meng/GYM-Work/dataset/ScienceQA/images/train/train_000000.png",
            question="Which of these states is farthest north?",
            choices=["West Virginia", "Louisiana", "Arizona", "Oklahoma"],
            max_rounds=2
        )
        
        print(f"\nMulti-round result: {multi_result['answer']}")
        print(f"Total bboxes: {len(multi_result['bboxes'])}")
        if multi_result['bboxes']:
            print(f"Bounding boxes: {multi_result['bboxes']}")
    else:
        print("\n⚠️ Model didn't generate the expected format")