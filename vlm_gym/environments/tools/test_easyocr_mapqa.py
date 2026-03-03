# test_easyocr_mapqa.py
import sys
import json
import os
from pathlib import Path
import time

try:
    import easyocr
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    print("EasyOCR import successful")
    
    # 创建reader（支持英文）
    print("Creating EasyOCR reader...")
    reader = easyocr.Reader(['en'])
    print("Reader created successfully")
    
    # 检查reader的方法
    print("\nAvailable methods in reader:")
    methods = [method for method in dir(reader) if not method.startswith('_')]
    print(methods[:20])
    
    # 加载MapQA数据
    data_path = "/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/MapQA/mapqa_train_vlmgym.json"
    
    print(f"\nLoading MapQA data from: {data_path}")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    # 获取唯一的图片列表（因为一张图片可能有多个问题）
    unique_images = {}
    for sample in data[:100]:  # 只看前100个样本
        img_path = sample['image_path']
        img_name = os.path.basename(img_path)
        if img_name not in unique_images:
            unique_images[img_name] = {
                'path': img_path,
                'questions': []
            }
        unique_images[img_name]['questions'].append({
            'id': sample['id'],
            'question': sample['question'],
            'answer': sample['answer'],
            'metadata': sample.get('metadata', {})
        })
    
    print(f"Unique images in first 100 samples: {len(unique_images)}")
    
    # 测试前3张不同的地图
    for idx, (img_name, img_info) in enumerate(list(unique_images.items())[:3]):
        print(f"\n{'='*80}")
        print(f"Image {idx + 1}: {img_name}")
        print(f"Number of questions for this image: {len(img_info['questions'])}")
        
        # 显示几个相关问题
        print("\nSample questions:")
        for i, q in enumerate(img_info['questions'][:3]):
            print(f"  Q{i+1}: {q['question']}")
            print(f"  A{i+1}: {q['answer']}")
        
        image_path = img_info['path']
        print(f"\nImage path: {image_path}")
        print(f"Image exists: {os.path.exists(image_path)}")
        
        if os.path.exists(image_path):
            # 加载图像
            img = Image.open(image_path)
            print(f"Image size: {img.size}")
            print(f"Image mode: {img.mode}")
            
            # 如果需要，转换为RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 使用EasyOCR进行文本检测和识别
            print(f"\nRunning OCR on {img_name}...")
            start_time = time.time()
            
            # 详细结果（包含边界框）
            result = reader.readtext(image_path)
            ocr_time = time.time() - start_time
            print(f"OCR completed in {ocr_time:.2f} seconds")
            print(f"Detected {len(result)} text regions")
            
            # 分析检测结果
            # 1. 收集所有州缩写
            state_abbreviations = []
            # 2. 收集数值范围
            value_ranges = []
            # 3. 收集其他文本
            other_texts = []
            
            # 美国州缩写集合
            us_state_abbrevs = {
                'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
                'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
                'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
                'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
                'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
            }
            
            print(f"\n{'='*40}")
            print("Detection Analysis:")
            
            for j, detection in enumerate(result):
                if len(detection) >= 3:
                    bbox, text, confidence = detection[0], detection[1], detection[2]
                    text_clean = text.strip().upper()
                    
                    # 检查是否是州缩写
                    if text_clean in us_state_abbrevs:
                        state_abbreviations.append({
                            'state': text_clean,
                            'confidence': confidence,
                            'bbox': bbox
                        })
                    # 检查是否包含数字（可能是范围）
                    elif any(char.isdigit() for char in text):
                        value_ranges.append({
                            'text': text,
                            'confidence': confidence,
                            'bbox': bbox
                        })
                    else:
                        other_texts.append({
                            'text': text,
                            'confidence': confidence,
                            'bbox': bbox
                        })
            
            # 打印分析结果
            print(f"\nDetected State Abbreviations: {len(state_abbreviations)}")
            if state_abbreviations:
                print("States found:")
                # 按置信度排序
                sorted_states = sorted(state_abbreviations, key=lambda x: x['confidence'], reverse=True)
                for state_info in sorted_states[:20]:  # 显示前20个
                    print(f"  - {state_info['state']} (confidence: {state_info['confidence']:.3f})")
            
            print(f"\nDetected Value Ranges: {len(value_ranges)}")
            if value_ranges:
                print("Values found:")
                sorted_values = sorted(value_ranges, key=lambda x: x['confidence'], reverse=True)
                for value_info in sorted_values[:10]:  # 显示前10个
                    print(f"  - '{value_info['text']}' (confidence: {value_info['confidence']:.3f})")
            
            print(f"\nOther Texts: {len(other_texts)}")
            if other_texts:
                print("Other texts found:")
                sorted_others = sorted(other_texts, key=lambda x: x['confidence'], reverse=True)
                for other_info in sorted_others[:10]:  # 显示前10个
                    print(f"  - '{other_info['text']}' (confidence: {other_info['confidence']:.3f})")
            
            # 保存带注释的图像
            try:
                output_dir = "/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/vlm_gym/environments/tools/easyocr_mapqa_output"
                os.makedirs(output_dir, exist_ok=True)
                
                # 画边界框
                img_annotated = img.copy()
                draw = ImageDraw.Draw(img_annotated)
                
                # 用不同颜色标注不同类型的文本
                # 红色：州缩写
                for state_info in state_abbreviations:
                    if state_info['confidence'] > 0.5:
                        bbox = state_info['bbox']
                        points = [(p[0], p[1]) for p in bbox]
                        for i in range(len(points)):
                            start = points[i]
                            end = points[(i + 1) % len(points)]
                            draw.line([start, end], fill='red', width=3)
                
                # 蓝色：数值
                for value_info in value_ranges:
                    if value_info['confidence'] > 0.5:
                        bbox = value_info['bbox']
                        points = [(p[0], p[1]) for p in bbox]
                        for i in range(len(points)):
                            start = points[i]
                            end = points[(i + 1) % len(points)]
                            draw.line([start, end], fill='blue', width=2)
                
                # 绿色：其他文本（标题等）
                for other_info in other_texts:
                    if other_info['confidence'] > 0.7 and len(other_info['text']) > 5:
                        bbox = other_info['bbox']
                        points = [(p[0], p[1]) for p in bbox]
                        for i in range(len(points)):
                            start = points[i]
                            end = points[(i + 1) % len(points)]
                            draw.line([start, end], fill='green', width=1)
                
                # 保存标注图像
                output_path = os.path.join(output_dir, f"annotated_{img_name}")
                img_annotated.save(output_path)
                print(f"\nAnnotated image saved to: {output_path}")
                
            except Exception as e:
                print(f"Error saving annotated image: {e}")
            
            # 检查OCR是否能帮助回答问题
            print(f"\n{'='*40}")
            print("Checking if OCR helps answer questions:")
            
            # 获取所有检测到的州名
            detected_states = [s['state'] for s in state_abbreviations]
            
            for q_idx, q_info in enumerate(img_info['questions'][:3]):
                print(f"\nQ: {q_info['question']}")
                print(f"Expected A: {q_info['answer']}")
                
                # 检查答案是否在检测结果中
                answer = str(q_info['answer']).upper()
                if any(state in answer for state in detected_states):
                    print("✓ OCR might help: detected states match answer")
                elif any(value['text'] in str(q_info['answer']) for value in value_ranges):
                    print("✓ OCR might help: detected values match answer")
                else:
                    print("✗ OCR results don't directly match answer")
            
        else:
            print(f"Image not found: {image_path}")
        
        print(f"\n" + "="*40)
    
    # 总结
    print(f"\n{'='*80}")
    print("Summary for MapQA with EasyOCR:")
    print("1. EasyOCR可以检测地图上的文本")
    print("2. 能够识别州缩写（如FL, TX, CA等）")
    print("3. 能够识别数值范围（如图例中的数字）")
    print("4. 可以识别标题和其他文本")
    print("\n潜在问题：")
    print("- 小字体的州缩写可能检测困难")
    print("- 白色文字在彩色背景上的对比度问题")
    print("- 数值范围的格式可能需要后处理")
    print("\n建议：")
    print("- 可以预处理图像以增强对比度")
    print("- 可以裁剪出特定区域（如图例）单独处理")
    print("- 结合颜色分析可能更有效")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()