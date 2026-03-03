# test_easyocr_real_data.py
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
    
    # 创建reader（支持英文和数学符号）
    reader = easyocr.Reader(['en'])  # 先用英文，因为数学公式主要是英文
    print("Reader created successfully")
    
    # 检查reader的方法
    print("\nAvailable methods in reader:")
    methods = [method for method in dir(reader) if not method.startswith('_')]
    print(methods[:20])  # 显示前20个方法
    
    # 加载真实数据
    data_path = "/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/olympiadbench/OlympiadBench_Dataset/combined_dataset/olympiadbench_format.json"
    image_dir = "/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/olympiadbench/OlympiadBench_Dataset/combined_dataset/images"
    
    print(f"\nLoading data from: {data_path}")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    # 测试前两个样本
    for i, sample in enumerate(data[:2]):
        print(f"\n{'='*60}")
        print(f"Sample {i}: {sample['id']}")
        print(f"Question preview: {sample['question'][:100]}...")
        print(f"Answer: {sample['answer']}")
        
        # 构建完整图片路径
        image_filename = sample['image_path'].split('/')[-1]  # 获取文件名
        image_path = os.path.join(image_dir, image_filename)
        
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
            print(f"\nRunning OCR on {image_filename}...")
            
            # 详细结果（包含边界框）
            result = reader.readtext(image_path)
            print(f"\nDetected {len(result)} text regions")
            
            # 显示前5个检测结果
            for j, detection in enumerate(result[:5]):
                if len(detection) >= 3:
                    bbox, text, confidence = detection[0], detection[1], detection[2]
                    print(f"\nDetection {j}:")
                    print(f"  - Text: '{text}'")
                    print(f"  - Confidence: {confidence:.3f}")
                    print(f"  - BBox: {bbox}")
            
            # 只获取文本（不含边界框）
            texts_only = reader.readtext(image_path, detail=0)
            print(f"\nAll detected texts: {texts_only}")
            
            # 保存带注释的图像（可选）
            try:
                # 创建输出目录
                output_dir = "/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/vlm_gym/environments/tools/easyocr_test_output"
                os.makedirs(output_dir, exist_ok=True)
                
                # 画边界框
                from PIL import ImageDraw, ImageFont
                img_annotated = img.copy()
                draw = ImageDraw.Draw(img_annotated)
                
                for detection in result[:10]:  # 只画前10个
                    bbox = detection[0]
                    text = detection[1]
                    confidence = detection[2]
                    
                    # 转换坐标格式
                    points = [(p[0], p[1]) for p in bbox]
                    
                    # 画边界框
                    for i in range(len(points)):
                        start = points[i]
                        end = points[(i + 1) % len(points)]
                        draw.line([start, end], fill='red', width=2)
                    
                    # 标注文本（在边界框上方）
                    if confidence > 0.5:  # 只标注高置信度的
                        draw.text(points[0], f"{text[:20]}... ({confidence:.2f})", fill='blue')
                
                # 保存标注图像
                output_path = os.path.join(output_dir, f"annotated_{image_filename}")
                img_annotated.save(output_path)
                print(f"\nAnnotated image saved to: {output_path}")
                
            except Exception as e:
                print(f"Error saving annotated image: {e}")
            
            # 分析数学相关内容
            print(f"\n{'='*40}")
            print("Math-related text detection:")
            math_symbols = ['=', '+', '-', '×', '÷', '∠', '°', 'π', '∞', '∑', '√', '∫']
            math_keywords = ['angle', 'triangle', 'circle', 'parallel', 'length', 'tan', 'sin', 'cos']
            
            detected_math = []
            for text in texts_only:
                text_lower = text.lower()
                # 检查数学符号
                if any(symbol in text for symbol in math_symbols):
                    detected_math.append(f"Symbol: {text}")
                # 检查数学关键词
                elif any(keyword in text_lower for keyword in math_keywords):
                    detected_math.append(f"Keyword: {text}")
                # 检查数字
                elif any(char.isdigit() for char in text):
                    detected_math.append(f"Number: {text}")
            
            print(f"Math-related detections: {len(detected_math)}")
            for item in detected_math[:10]:  # 显示前10个
                print(f"  - {item}")
            
        else:
            print(f"Image not found: {image_path}")
    
    # 测试其他功能
    print(f"\n{'='*60}")
    print("Testing additional EasyOCR features:")
    
    # 检查支持的语言
    print("\nChecking language support...")
    # 注意：不是所有版本都有get_available_languages
    if hasattr(reader, 'lang_list'):
        print(f"Current languages: {reader.lang_list}")
    
    # 测试批处理能力
    if len(data) >= 2 and all(os.path.exists(os.path.join(image_dir, d['image_path'].split('/')[-1])) for d in data[:2]):
        print("\nTesting batch processing...")
        image_paths = [os.path.join(image_dir, d['image_path'].split('/')[-1]) for d in data[:2]]
        
        try:
            # 一些版本支持批处理
            batch_results = reader.readtext(image_paths[0])  # EasyOCR通常不支持真正的批处理
            print("Note: EasyOCR processes images one by one")
        except Exception as e:
            print(f"Batch processing note: {e}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# 总结
print(f"\n{'='*60}")
print("Summary:")
print("1. EasyOCR可以检测和识别图像中的文本")
print("2. 返回格式：[(bbox, text, confidence), ...]")
print("3. bbox是4个点的坐标：[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]")
print("4. 可以设置detail=0只获取文本列表")
print("\n注意事项：")
print("- EasyOCR主要针对印刷体文本，手写体效果可能较差")
print("- 对于数学公式，可能需要后处理来组合检测到的符号")
print("- 复杂的数学公式（如分数、上下标）可能需要专门的工具")