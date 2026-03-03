# test_easyocr_chartqa.py
import sys
sys.path.append('/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/vlm_gym/environments/tools/EasyOCR')

import json
import os
from pathlib import Path

try:
    import easyocr
    import numpy as np
    from PIL import Image
    print("EasyOCR import successful")
    
    # 创建reader（支持英文和数字）
    reader = easyocr.Reader(['en'])  # ChartQA主要是英文内容
    print("Reader created successfully")
    
    # 检查reader的方法
    print("\nAvailable methods in reader:")
    methods = [method for method in dir(reader) if not method.startswith('_')]
    print(methods[:20])  # 显示前20个方法
    
    # 加载ChartQA数据
    data_path = "/workspace/mathvista/data/chartqa/converted_train/train_human_vlmgym_container.json"
    
    print(f"\nLoading data from: {data_path}")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    # 测试前三个样本
    for i, sample in enumerate(data[:3]):
        print(f"\n{'='*60}")
        print(f"Sample {i}: {sample['id']}")
        print(f"Question: {sample['question']}")
        print(f"Answer: {sample['answer']}")
        print(f"Task: {sample.get('task', 'N/A')}")
        
        # 使用完整的图片路径
        image_path = sample['image_path']
        
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
            print(f"\nRunning OCR on {os.path.basename(image_path)}...")
            
            # 详细结果（包含边界框）
            result = reader.readtext(image_path)
            print(f"\nDetected {len(result)} text regions")
            
            # 显示前10个检测结果
            for j, detection in enumerate(result[:10]):
                if len(detection) >= 3:
                    bbox, text, confidence = detection[0], detection[1], detection[2]
                    print(f"\nDetection {j}:")
                    print(f"  - Text: '{text}'")
                    print(f"  - Confidence: {confidence:.3f}")
                    print(f"  - BBox: {bbox}")
            
            # 只获取文本（不含边界框）
            texts_only = reader.readtext(image_path, detail=0)
            print(f"\nAll detected texts ({len(texts_only)} total):")
            # 显示前20个文本
            for j, text in enumerate(texts_only[:20]):
                print(f"  {j}: '{text}'")
            if len(texts_only) > 20:
                print(f"  ... and {len(texts_only) - 20} more texts")
            
            # 保存带注释的图像（可选）
            try:
                # 创建输出目录
                output_dir = "/workspace/mathvista/easyocr_test_output"
                os.makedirs(output_dir, exist_ok=True)
                
                # 画边界框
                from PIL import ImageDraw, ImageFont
                img_annotated = img.copy()
                draw = ImageDraw.Draw(img_annotated)
                
                # 为不同置信度使用不同颜色
                for detection in result[:20]:  # 只画前20个
                    bbox = detection[0]
                    text = detection[1]
                    confidence = detection[2]
                    
                    # 根据置信度选择颜色
                    if confidence > 0.9:
                        color = 'green'
                    elif confidence > 0.7:
                        color = 'orange'
                    else:
                        color = 'red'
                    
                    # 转换坐标格式
                    points = [(p[0], p[1]) for p in bbox]
                    
                    # 画边界框
                    for k in range(len(points)):
                        start = points[k]
                        end = points[(k + 1) % len(points)]
                        draw.line([start, end], fill=color, width=2)
                    
                    # 标注文本（在边界框上方）
                    if confidence > 0.5:  # 只标注高置信度的
                        text_display = text[:15] + "..." if len(text) > 15 else text
                        draw.text(points[0], f"{text_display} ({confidence:.2f})", fill='blue')
                
                # 保存标注图像
                image_filename = os.path.basename(image_path)
                output_path = os.path.join(output_dir, f"annotated_{image_filename}")
                img_annotated.save(output_path)
                print(f"\nAnnotated image saved to: {output_path}")
                
            except Exception as e:
                print(f"Error saving annotated image: {e}")
            
            # 分析图表相关内容
            print(f"\n{'='*40}")
            print("Chart-related text analysis:")
            
            # 检测数字
            numbers = [text for text in texts_only if any(char.isdigit() for char in text)]
            print(f"\nNumbers detected ({len(numbers)}):")
            for num in numbers[:10]:
                print(f"  - {num}")
            
            # 检测年份（4位数字）
            years = [text for text in texts_only if text.isdigit() and len(text) == 4 and 1900 <= int(text) <= 2100]
            if years:
                print(f"\nPossible years: {years}")
            
            # 检测百分比
            percentages = [text for text in texts_only if '%' in text or 'percent' in text.lower()]
            if percentages:
                print(f"\nPercentages: {percentages}")
            
            # 检测标签和标题（通常是较长的文本）
            labels = [text for text in texts_only if len(text) > 10 and not text.isdigit()]
            print(f"\nPossible labels/titles ({len(labels)}):")
            for label in labels[:5]:
                print(f"  - {label}")
            
        else:
            print(f"Image not found: {image_path}")
    
    # 统计分析
    print(f"\n{'='*60}")
    print("Dataset Statistics:")
    
    # 统计每个图片出现的次数
    image_counts = {}
    for sample in data[:100]:  # 分析前100个样本
        img_name = os.path.basename(sample['image_path'])
        image_counts[img_name] = image_counts.get(img_name, 0) + 1
    
    # 找出重复使用的图片
    repeated_images = {k: v for k, v in image_counts.items() if v > 1}
    if repeated_images:
        print(f"\nImages used multiple times (in first 100 samples):")
        for img, count in sorted(repeated_images.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  - {img}: {count} times")
    
    # 测试其他功能
    print(f"\n{'='*60}")
    print("Testing additional EasyOCR features:")
    
    # 检查支持的语言
    print("\nChecking language support...")
    if hasattr(reader, 'lang_list'):
        print(f"Current languages: {reader.lang_list}")
    
    # 测试不同的参数设置
    if data and os.path.exists(data[0]['image_path']):
        print("\nTesting different parameter settings...")
        test_image = data[0]['image_path']
        
        # 尝试段落模式
        try:
            paragraph_result = reader.readtext(test_image, paragraph=True)
            print(f"Paragraph mode: Detected {len(paragraph_result)} text regions")
        except Exception as e:
            print(f"Paragraph mode error: {e}")
        
        # 尝试调整阈值
        try:
            low_threshold_result = reader.readtext(test_image, text_threshold=0.5, low_text=0.3)
            print(f"Lower threshold: Detected {len(low_threshold_result)} text regions")
        except Exception as e:
            print(f"Threshold adjustment error: {e}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# 总结
print(f"\n{'='*60}")
print("Summary:")
print("1. EasyOCR可以检测和识别图表中的文本")
print("2. 对于ChartQA数据集，主要识别：")
print("   - 数字和数值")
print("   - 年份")
print("   - 百分比")
print("   - 图表标题和标签")
print("   - 图例文本")
print("\n注意事项：")
print("- 图表中的文本方向可能多样（水平、垂直、倾斜）")
print("- 小字体或密集文本可能需要调整参数")
print("- 图表背景可能影响文本识别")
print("- 可能需要后处理来关联文本与图表元素")