#最新的让qwen生成bbox然后调用deepeyes进行图片放大的脚本
import re
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image, ImageDraw
import torch
import os
import json
from typing import Dict, List, Optional
import random
import argparse
import sys

# 添加DeepEyes工具路径
sys.path.append('/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista')
from vlm_gym.environments.tools.deepeyes_tool import DeepEyesTool

# 定义路径
OUTPUT_DIR = "/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/vlm_gym/environments/tools/crop_image/"
DATASET_PATH = "/data/wang/meng/GYM-Work/dataset/ScienceQA/reformatted/train.json"
IMAGE_BASE_PATH = "/data/wang/meng/GYM-Work/dataset/ScienceQA/"

class GroundR1_ScienceQA_WithDeepEyes:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
        """初始化模型、数据集和DeepEyes工具"""
        # 确保输出目录存在
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"📁 Output directory: {OUTPUT_DIR}")
        
        # 加载模型
        print(f"🔄 Loading model: {model_name}")
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # 加载数据集
        print(f"📚 Loading dataset from: {DATASET_PATH}")
        with open(DATASET_PATH, 'r') as f:
            self.dataset = json.load(f)
        print(f"✅ Loaded {len(self.dataset)} examples")
        
        # 初始化DeepEyes工具
        print(f"🔧 Initializing DeepEyes tool...")
        self.deepeyes_tool = DeepEyesTool()
        print(f"✅ DeepEyes tool initialized")
    
    def construct_prompt_with_context(self, question: str, choices: List[str], 
                                    hint: Optional[str] = None, 
                                    lecture: Optional[str] = None) -> str:
        """构造包含hint和lecture的prompt"""
        # 给选项添加标签
        labeled_choices = [f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)]
        choice_text = "\n".join(labeled_choices)
        
        # 构建上下文部分
        context_parts = []
        if hint:
            context_parts.append(f"Background Information:\n{hint}")
        if lecture:
            context_parts.append(f"Related Knowledge:\n{lecture}")
        
        context_text = "\n\n".join(context_parts) if context_parts else ""
        
        # 构建完整prompt
        prompt = f"""<|im_start|>system
You are a helpful assistant that analyzes images and answers questions step by step.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>{context_text}

Question: {question}

Options:
{choice_text}

Please analyze this step by step:
1. First, describe your reasoning in <think> tags
2. If you need to focus on a specific region, provide coordinates in <box>[x1,y1,x2,y2]</box> format
3. Finally, provide your answer (just the letter, e.g., "A") in <answer> tags

Example format:
<think>I need to analyze the image carefully...</think>
<box>[100,200,300,400]</box>
<answer>A</answer>

Now begin your analysis:<|im_end|>
<|im_start|>assistant
<think>"""
        
        return prompt
    
    def fix_bbox_format(self, text):
        """修复bbox格式"""
        return re.sub(r'<box\[(\d+),(\d+),(\d+),(\d+)\]></box>', r'<box>[\1,\2,\3,\4]</box>', text)
    
    def parse_response(self, text):
        """解析模型响应"""
        # 分离实际生成的内容
        if "assistant" in text:
            parts = text.split("assistant")
            if len(parts) > 1:
                text = parts[-1]
        
        # 修复bbox格式
        text = self.fix_bbox_format(text)
        
        results = {
            'thinking': None,
            'bbox': None,
            'answer': None,
            'raw_response': text
        }
        
        # 提取thinking
        think_pattern = r'<think>(.*?)</think>'
        think_matches = re.findall(think_pattern, text, re.DOTALL)
        if think_matches:
            real_thinks = [t.strip() for t in think_matches if t.strip() != "reasoning"]
            results['thinking'] = real_thinks[-1] if real_thinks else None
        
        # 提取bbox
        bbox_pattern = r'<box>\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*</box>'
        bbox_match = re.search(bbox_pattern, text)
        if bbox_match:
            results['bbox'] = [int(x) for x in bbox_match.groups()]
        
        # 提取answer
        answer_pattern = r'<answer>(.*?)</answer>'
        answer_matches = re.findall(answer_pattern, text, re.DOTALL)
        if answer_matches:
            real_answers = [a.strip() for a in answer_matches if a.strip() != "answer"]
            results['answer'] = real_answers[-1] if real_answers else None
        
        return results
    
    def use_deepeyes_on_bbox(self, image: Image.Image, bbox: List[int], example_id: str, 
                           question: str, choices: List[str], hint: str = None, lecture: str = None) -> Dict:
        """使用DeepEyes工具对bbox区域进行分析"""
        print(f"\n🔍 Using DeepEyes tool on bbox {bbox}...")
        
        # 重置DeepEyes工具
        self.deepeyes_tool.reset(image)
        
        # 使用bbox进行缩放
        zoom_action = f"""
        <tool_call>
        {{"name": "image_zoom_in_tool", "arguments": {{"bbox_2d": {bbox}}}}}
        </tool_call>
        """
        
        zoom_result = self.deepeyes_tool.execute(zoom_action)
        
        if zoom_result.get('success'):
            # 保存DeepEyes缩放后的图像
            deepeyes_image = zoom_result['image'] #这里获取的是缩放后的图像
            deepeyes_path = os.path.join(OUTPUT_DIR, f"example_{example_id}_deepeyes_zoom.png")
            deepeyes_image.save(deepeyes_path)
            print(f"📸 Saved DeepEyes zoomed image to: {deepeyes_path}")
            
            # 使用Qwen模型对DeepEyes缩放后的图像进行重新分析
            print(f"🔄 Re-analyzing zoomed image with Qwen model...")
            
            # 构建新的prompt，使用缩放后的图像
            prompt = self.construct_prompt_with_context(
                question=question,
                choices=choices,
                hint=hint,
                lecture=lecture
            )
            
            # 使用Qwen模型对缩放后的图像进行重新推理分析
            inputs = self.processor(
                text=prompt,
                images=deepeyes_image,  # 这里传入的是使用DeepEyes缩放后的图像
                return_tensors="pt"
            ).to(self.model.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.2,
                    do_sample=True
                )
            
            response = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            
            # 解析响应
            if "<think>" in response:
                response = response.split("<think>", 1)[1]
                response = "<think>" + response
            
            parsed = self.parse_response(response)
            
            print(f"🔧 DeepEyes + Qwen Response: {response[:200]}...")
            
            return {
                'success': True,
                'zoomed_image': deepeyes_image,
                'deepeyes_answer': parsed['answer'],
                'deepeyes_thinking': parsed['thinking'],
                'full_result': parsed
            }
        else:
            return {
                'success': False,
                'error': 'Failed to zoom with DeepEyes'
            }
    
    def process_single_example(self, example_id: str, save_images: bool = True):
        """处理单个样例，对错误的答案使用DeepEyes工具"""
        example = self.dataset[example_id]
        
        print(f"\n{'='*80}")
        print(f"📝 Processing Example ID: {example_id}")
        print(f"❓ Question: {example['question']}")
        print(f"🎯 Ground Truth: {example['answer']}")
        
        # 检查是否有图像
        if not example.get('image'):
            print(f"⚠️ No image for this example. Skipping...")
            return {
                'example_id': example_id,
                'question': example['question'],
                'ground_truth': example['answer'],
                'predicted': None,
                'bbox': None,
                'thinking': None,
                'is_correct': False,
                'error': 'No image available'
            }
        
        # 构建图像路径
        image_path = os.path.join(IMAGE_BASE_PATH, example['image'])
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return {
                'example_id': example_id,
                'question': example['question'],
                'ground_truth': example['answer'],
                'predicted': None,
                'bbox': None,
                'thinking': None,
                'is_correct': False,
                'error': f'Image file not found: {image_path}'
            }
        
        # 加载图像
        image = Image.open(image_path)
        print(f"📐 Image size: {image.size}")
        
        # 构建prompt
        prompt = self.construct_prompt_with_context(
            question=example['question'],
            choices=example['choices'],
            hint=example.get('hint'),
            lecture=example.get('lecture')
        )
        
        # 执行推理
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2,
                do_sample=True
            )
        
        response = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        
        # 提取生成的部分
        if "<think>" in response:
            response = response.split("<think>", 1)[1]
            response = "<think>" + response
        
        print(f"\n🤖 Model Response:\n{response[:500]}...")
        
        # 解析响应
        parsed = self.parse_response(response)
        
        print(f"\n📊 Parsed Results:")
        if parsed['thinking']:
            print(f"💭 Thinking: {parsed['thinking'][:200]}...")
        print(f"📦 BBox: {parsed['bbox']}")
        print(f"✅ Model Answer: {parsed['answer']}")
        
        # 检查答案是否正确
        is_correct = False
        predicted_choice = None
        deepeyes_result = None
        
        if parsed['answer'] and example.get('choices'):
            # 对于多选题，将答案字母映射到实际选项
            answer_mapping = {chr(65+i): choice for i, choice in enumerate(example['choices'])}
            predicted_choice = answer_mapping.get(parsed['answer'].upper())
            is_correct = predicted_choice == example['answer']
        elif parsed['answer'] and not example.get('choices'):
            # 对于非多选题，直接比较答案
            is_correct = parsed['answer'].lower() == example['answer'].lower()
            predicted_choice = parsed['answer']
        
        if parsed['answer']:
            print(f"🎯 Correct: {is_correct}")
        
        # 如果答案错误且有bbox，使用DeepEyes工具进行进一步分析
        if not is_correct and parsed['bbox']:
            print(f"\n❌ Answer is incorrect and bbox is available. Using DeepEyes for further analysis...")
            deepeyes_result = self.use_deepeyes_on_bbox(
                image, 
                parsed['bbox'], 
                example_id,
                example['question'],
                example['choices'],
                example.get('hint'),
                example.get('lecture')
            )
            
            if deepeyes_result['success'] and deepeyes_result['deepeyes_answer']:
                print(f"🔧 DeepEyes + Qwen Answer: {deepeyes_result['deepeyes_answer']}")
                if deepeyes_result.get('deepeyes_thinking'):
                    print(f"🔧 DeepEyes + Qwen Thinking: {deepeyes_result['deepeyes_thinking'][:200]}...")
                
                # 检查DeepEyes的答案是否正确
                if example.get('choices'):
                    answer_mapping = {chr(65+i): choice for i, choice in enumerate(example['choices'])}
                    deepeyes_predicted = answer_mapping.get(deepeyes_result['deepeyes_answer'].upper())
                    deepeyes_correct = deepeyes_predicted == example['answer']
                    print(f"🔧 DeepEyes + Qwen Correct: {deepeyes_correct}")
        
        # 保存可视化
        if save_images and parsed['bbox']:
            try:
                # 保存带bbox的图像
                img_copy = image.copy()
                draw = ImageDraw.Draw(img_copy)
                draw.rectangle(parsed['bbox'], outline='red', width=3)
                
                bbox_path = os.path.join(OUTPUT_DIR, f"example_{example_id}_bbox.png")
                img_copy.save(bbox_path)
                print(f"📸 Saved bbox visualization to: {bbox_path}")
                
                # 保存裁剪的图像
                x1, y1, x2, y2 = parsed['bbox']
                cropped = image.crop((x1, y1, x2, y2))
                cropped_path = os.path.join(OUTPUT_DIR, f"example_{example_id}_cropped.png")
                cropped.save(cropped_path)
                print(f"📸 Saved cropped region to: {cropped_path}")
            except Exception as e:
                print(f"⚠️ Error saving images: {e}")
        
        result = {
            'example_id': example_id,
            'question': example['question'],
            'ground_truth': example['answer'],
            'predicted': predicted_choice if predicted_choice else parsed['answer'],
            'bbox': parsed['bbox'],
            'thinking': parsed['thinking'],
            'is_correct': is_correct,
            'has_image': True
        }
        
        # 添加DeepEyes结果（如果有）
        if deepeyes_result:
            result['deepeyes_analysis'] = {
                'used': True,
                'success': deepeyes_result['success'],
                'answer': deepeyes_result.get('deepeyes_answer'),
                'thinking': deepeyes_result.get('deepeyes_thinking'),
                'correct': deepeyes_correct if 'deepeyes_correct' in locals() else None
            }
        else:
            result['deepeyes_analysis'] = {'used': False}
        
        return result
    
    def process_multiple_examples(self, num_examples: int = 5, random_sample: bool = True, 
                                 example_ids: List[str] = None, save_images: bool = True):
        """处理多个样例"""
        # 选择样例
        if example_ids is None:
            if random_sample:
                example_ids = random.sample(list(self.dataset.keys()), min(num_examples, len(self.dataset)))
            else:
                example_ids = list(self.dataset.keys())[:num_examples]
        
        results = []
        correct_count = 0
        skipped_count = 0
        deepeyes_used_count = 0
        deepeyes_correct_count = 0
        
        for i, example_id in enumerate(example_ids):
            print(f"\n[{i+1}/{len(example_ids)}] Processing example {example_id}...")
            result = self.process_single_example(example_id, save_images=save_images)
            if result:
                results.append(result)
                if result.get('error'):
                    skipped_count += 1
                elif result['is_correct']:
                    correct_count += 1
                
                # 统计DeepEyes使用情况
                if result.get('deepeyes_analysis', {}).get('used'):
                    deepeyes_used_count += 1
                    if result['deepeyes_analysis'].get('correct'):
                        deepeyes_correct_count += 1
        
        # 打印总结
        print(f"\n{'='*80}")
        print(f"📊 Summary:")
        print(f"Total attempted: {len(example_ids)}")
        print(f"Successfully processed: {len(results) - skipped_count}")
        print(f"Skipped (no image): {skipped_count}")
        
        processed_count = len(results) - skipped_count
        if processed_count > 0:
            print(f"\n🤖 Qwen Model Performance:")
            print(f"   Correct: {correct_count}")
            print(f"   Accuracy: {correct_count/processed_count*100:.2f}%")
            
            if deepeyes_used_count > 0:
                print(f"\n🔧 DeepEyes + Qwen Re-analysis:")
                print(f"   Used on: {deepeyes_used_count} incorrect examples with bbox")
                print(f"   Corrected: {deepeyes_correct_count}")
                print(f"   Re-analysis Success Rate: {deepeyes_correct_count/deepeyes_used_count*100:.2f}%")
                print(f"   Overall accuracy after re-analysis: {(correct_count + deepeyes_correct_count)/processed_count*100:.2f}%")
        
        return results

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='Ground-R1 for ScienceQA with Qwen2.5-VL and DeepEyes integration')
    parser.add_argument('--num_examples', type=int, default=5, 
                       help='Number of examples to process (default: 5)')
    parser.add_argument('--random', action='store_true', default=True,
                       help='Random sampling of examples (default: True)')
    parser.add_argument('--sequential', action='store_true',
                       help='Process examples sequentially instead of randomly')
    parser.add_argument('--start_idx', type=str, default=None,
                       help='Start from a specific example ID (only works with --sequential)')
    parser.add_argument('--single_id', type=str, default=None,
                       help='Process only a single specific example ID')
    parser.add_argument('--no_save_images', action='store_true',
                       help='Do not save visualization images')
    parser.add_argument('--only_with_images', action='store_true',
                       help='Only process examples that have images')
    parser.add_argument('--output_file', type=str, 
                       default=os.path.join(OUTPUT_DIR, "ground_r1_deepeyes_results.json"),
                       help='Output JSON file path')
    
    args = parser.parse_args()
    
    # 初始化
    print("🚀 Initializing Ground-R1 with DeepEyes for ScienceQA...")
    grounding = GroundR1_ScienceQA_WithDeepEyes()
    
    # 如果只处理有图像的样例，过滤数据集
    if args.only_with_images:
        original_size = len(grounding.dataset)
        grounding.dataset = {k: v for k, v in grounding.dataset.items() if v.get('image')}
        filtered_size = len(grounding.dataset)
        print(f"📷 Filtered to examples with images: {filtered_size}/{original_size}")
    
    save_images = not args.no_save_images
    results = []
    
    # 处理单个特定样例
    if args.single_id:
        print(f"\n🔍 Processing single example ID: {args.single_id}")
        if args.single_id not in grounding.dataset:
            print(f"❌ Example ID '{args.single_id}' not found in dataset")
            return
        result = grounding.process_single_example(args.single_id, save_images=save_images)
        results = [result] if result else []
    else:
        # 处理多个样例
        random_sample = not args.sequential
        print(f"\n🔄 Processing {args.num_examples} examples ({'random' if random_sample else 'sequential'} sampling)...")
        
        example_ids = None
        if args.start_idx and args.sequential:
            # 如果指定了起始ID且是顺序处理
            all_ids = list(grounding.dataset.keys())
            try:
                start_position = all_ids.index(args.start_idx)
                example_ids = all_ids[start_position:start_position + args.num_examples]
                print(f"📍 Starting from ID '{args.start_idx}' (position {start_position})")
            except ValueError:
                print(f"⚠️ Start ID '{args.start_idx}' not found. Using first {args.num_examples} examples.")
                example_ids = all_ids[:args.num_examples]
        
        results = grounding.process_multiple_examples(
            num_examples=args.num_examples, 
            random_sample=random_sample,
            example_ids=example_ids,
            save_images=save_images
        )
    
    # 保存结果
    if results:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {args.output_file}")
    else:
        print("\n⚠️ No results to save.")

if __name__ == "__main__":
    main()