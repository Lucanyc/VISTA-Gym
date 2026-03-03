# test_sympy_on_unigeo_fixed.py
import sys
sys.path.append('/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista')

import json
import os
import re
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches

try:
    # 导入SymPy几何工具
    from vlm_gym.environments.tools.geometry_tools.sympy_geometry import SympyGeometryTool
    print("SympyGeometryTool import successful")
    
    # 创建SymPy工具实例
    sympy_config = {
        'precision': 4  # 设置精度为4位小数
    }
    
    sympy_tool = SympyGeometryTool(sympy_config)
    print("SympyGeometryTool created successfully")
    
    # 检查工具能力
    capabilities = sympy_tool.get_capabilities()
    print("\nSymPy Geometry Capabilities:")
    print(f"- Name: {capabilities['name']}")
    print(f"- Description: {capabilities['description']}")
    print(f"- Capabilities: {capabilities['capabilities']}")
    print(f"- Available functions: {[f['name'] for f in capabilities['functions']]}")
    
    # 加载UniGeo数据
    data_path = "/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/UniGeo/unigeo_train_vlmgym.json"
    
    print(f"\nLoading data from: {data_path}")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    # 修改输出目录到你有权限的位置
    output_dir = "/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/vlm_gym/environments/tools/sympy_unigeo_output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # 重置工具
    sympy_tool.reset()
    
    # 测试前10个样本
    for i, sample in enumerate(data[:10]):
        print(f"\n{'='*60}")
        print(f"Sample {i+1}: {sample.get('id', 'N/A')}")
        print(f"Question: {sample.get('question', 'N/A')[:100]}...")
        print(f"Answer: {sample.get('answer', 'N/A')}")
        
        # 获取图片路径（如果有）
        image_path = sample.get('image_path', '')
        if image_path:
            print(f"\nImage path: {image_path}")
            print(f"Image exists: {os.path.exists(image_path)}")
        
        # 分析问题类型并应用相应的几何计算
        question = sample.get('question', '').lower()
        
        # 策略1: 三角形角度计算
        if any(keyword in question for keyword in ['angle', 'triangle', '角', '三角形']):
            print("\n[Strategy 1] Triangle Angle Calculation")
            
            # 示例：计算一个三角形的角度
            # 这里使用默认坐标，实际应用中需要从问题或图像中提取
            triangle_action = {
                "function": "triangle_angle",
                "args": {
                    "A": [0, 0],
                    "B": [3, 0],
                    "C": [0, 4],
                    "vertex": "A"
                }
            }
            
            result = sympy_tool.execute(json.dumps(triangle_action))
            
            if result.get('success'):
                print(f"✓ Triangle angle calculation successful")
                print(f"  - Result: {result.get('formatted')}")
                print(f"  - Angle at vertex A: {result.get('result', {}).get('angle_degrees')}°")
            else:
                print(f"✗ Calculation failed: {result.get('error')}")
            
            # 计算所有角度
            print("\n  Computing all angles:")
            all_angles_action = {
                "function": "triangle_all_angles",
                "args": {
                    "A": [0, 0],
                    "B": [3, 0],
                    "C": [0, 4]
                }
            }
            
            all_result = sympy_tool.execute(json.dumps(all_angles_action))
            if all_result.get('success'):
                angles = all_result.get('result', {})
                print(f"  - Angle A: {angles.get('angle_A', 'N/A')}°")
                print(f"  - Angle B: {angles.get('angle_B', 'N/A')}°")
                print(f"  - Angle C: {angles.get('angle_C', 'N/A')}°")
                print(f"  - Sum: {angles.get('angle_sum', 'N/A')}°")
        
        # 策略2: 勾股定理
        if any(keyword in question for keyword in ['pythagorean', 'right triangle', '勾股', '直角三角形']):
            print("\n[Strategy 2] Pythagorean Theorem")
            
            # 从问题中提取数字（简单示例）
            numbers = re.findall(r'\d+', question)
            if len(numbers) >= 2:
                a, b = float(numbers[0]), float(numbers[1])
                pythagorean_action = {
                    "function": "pythagorean",
                    "args": {
                        "a": a,
                        "b": b
                    }
                }
                
                result = sympy_tool.execute(json.dumps(pythagorean_action))
                
                if result.get('success'):
                    print(f"✓ Pythagorean calculation successful")
                    print(f"  - Given: a={a}, b={b}")
                    print(f"  - Result: c={result.get('result', {}).get('c')}")
            else:
                # 使用默认值演示
                pythagorean_action = {
                    "function": "pythagorean",
                    "args": {"a": 3, "b": 4}
                }
                result = sympy_tool.execute(json.dumps(pythagorean_action))
                if result.get('success'):
                    print(f"✓ Pythagorean (default): {result.get('formatted')}")
        
        # 策略3: 圆的计算
        if any(keyword in question for keyword in ['circle', 'radius', 'circumference', '圆', '半径', '周长']):
            print("\n[Strategy 3] Circle Calculations")
            
            # 通过三点确定圆
            circle_action = {
                "function": "circle_from_points",
                "args": {
                    "p1": [0, 0],
                    "p2": [3, 0],
                    "p3": [0, 3]
                }
            }
            
            result = sympy_tool.execute(json.dumps(circle_action))
            
            if result.get('success'):
                circle = result.get('result', {})
                print(f"✓ Circle calculation successful")
                print(f"  - Center: {circle.get('center')}")
                print(f"  - Radius: {circle.get('radius')}")
                print(f"  - Area: {circle.get('area')}")
                print(f"  - Circumference: {circle.get('circumference')}")
            
            # 圆周角计算
            if 'inscribed' in question or '圆周角' in question:
                inscribed_action = {
                    "function": "inscribed_angle",
                    "args": {
                        "center": [0, 0],
                        "radius": 5,
                        "angle_degrees": 30
                    }
                }
                
                result = sympy_tool.execute(json.dumps(inscribed_action))
                if result.get('success'):
                    angle_result = result.get('result', {})
                    print(f"\n  Inscribed angle theorem:")
                    print(f"  - Inscribed angle: {angle_result.get('inscribed_angle')}°")
                    print(f"  - Central angle: {angle_result.get('central_angle')}°")
        
        # 策略4: 多边形面积
        if any(keyword in question for keyword in ['polygon', 'area', 'square', 'rectangle', '多边形', '面积']):
            print("\n[Strategy 4] Polygon Area Calculation")
            
            # 矩形示例
            if 'rectangle' in question or '矩形' in question:
                polygon_action = {
                    "function": "polygon_area",
                    "args": {
                        "vertices": [[0, 0], [4, 0], [4, 3], [0, 3]]
                    }
                }
            else:
                # 五边形示例
                polygon_action = {
                    "function": "polygon_area",
                    "args": {
                        "vertices": [[0, 0], [2, 0], [3, 1.5], [1, 2.5], [-1, 1]]
                    }
                }
            
            result = sympy_tool.execute(json.dumps(polygon_action))
            
            if result.get('success'):
                poly_result = result.get('result', {})
                print(f"✓ Polygon calculation successful")
                print(f"  - Area: {poly_result.get('area')}")
                print(f"  - Perimeter: {poly_result.get('perimeter')}")
                print(f"  - Number of vertices: {poly_result.get('num_vertices')}")
                print(f"  - Is convex: {poly_result.get('is_convex')}")
        
        # 策略5: 线段夹角
        if any(keyword in question for keyword in ['angle between', 'perpendicular', 'parallel', '夹角', '垂直', '平行']):
            print("\n[Strategy 5] Angle Between Lines")
            
            angle_action = {
                "function": "angle_between_lines",
                "args": {
                    "p1": [0, 0],
                    "p2": [1, 0],
                    "p3": [0, 0],
                    "p4": [1, 1]
                }
            }
            
            result = sympy_tool.execute(json.dumps(angle_action))
            
            if result.get('success'):
                angle_result = result.get('result', {})
                print(f"✓ Line angle calculation successful")
                print(f"  - Angle: {angle_result.get('angle_degrees')}°")
                print(f"  - Is parallel: {angle_result.get('is_parallel')}")
                print(f"  - Is perpendicular: {angle_result.get('is_perpendicular')}")
        
        # 策略6: 已知两角求第三角
        if 'third angle' in question or '第三个角' in question:
            print("\n[Strategy 6] Triangle Third Angle")
            
            # 尝试从问题中提取角度
            angles = re.findall(r'(\d+)°', question)
            if len(angles) >= 2:
                angle1, angle2 = float(angles[0]), float(angles[1])
                third_angle_action = {
                    "function": "triangle_angle_from_two",
                    "args": {
                        "angle1": angle1,
                        "angle2": angle2
                    }
                }
                
                result = sympy_tool.execute(json.dumps(third_angle_action))
                
                if result.get('success'):
                    angle_result = result.get('result', {})
                    print(f"✓ Third angle calculation successful")
                    print(f"  - Given angles: {angle1}°, {angle2}°")
                    print(f"  - Third angle: {angle_result.get('angle3')}°")
        
        # 创建可视化（如果有图像）
        if image_path and os.path.exists(image_path):
            print("\n[Creating Visualization]")
            try:
                img = Image.open(image_path)
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                
                ax.imshow(img)
                ax.set_title(f"Sample: {sample.get('id', 'N/A')}\nAnswer: {sample.get('answer', 'N/A')}")
                ax.axis('off')
                
                # 添加问题文本（截断长文本）
                question_text = sample.get('question', '')
                if len(question_text) > 100:
                    question_text = question_text[:100] + "..."
                plt.figtext(0.5, 0.02, f"Q: {question_text}", 
                           ha='center', fontsize=10, wrap=True)
                
                # 保存图像
                output_path = os.path.join(output_dir, f"sample_{sample.get('id', i)}.png")
                plt.tight_layout()
                plt.savefig(output_path, bbox_inches='tight', dpi=150)
                plt.close()
                print(f"✓ Visualization saved to: {output_path}")
                
            except Exception as e:
                print(f"✗ Error creating visualization: {e}")
    
    # 统计分析
    print(f"\n{'='*60}")
    print("Dataset Analysis with SymPy:")
    
    # 分析问题类型分布
    question_types = {
        'triangle': 0,
        'circle': 0,
        'angle': 0,
        'area': 0,
        'pythagorean': 0,
        'polygon': 0,
        'other': 0
    }
    
    print("\nAnalyzing first 100 samples for question types...")
    for i, sample in enumerate(data[:100]):
        if i % 20 == 0:
            print(f"  Processing sample {i}...")
        
        question_lower = sample.get('question', '').lower()
        
        # 统计问题类型
        if 'triangle' in question_lower or '三角形' in question_lower:
            question_types['triangle'] += 1
        elif 'circle' in question_lower or '圆' in question_lower:
            question_types['circle'] += 1
        elif 'angle' in question_lower or '角' in question_lower:
            question_types['angle'] += 1
        elif 'area' in question_lower or '面积' in question_lower:
            question_types['area'] += 1
        elif 'pythagorean' in question_lower or '勾股' in question_lower:
            question_types['pythagorean'] += 1
        elif 'polygon' in question_lower or '多边形' in question_lower:
            question_types['polygon'] += 1
        else:
            question_types['other'] += 1
    
    print("\nQuestion type distribution:")
    for qtype, count in sorted(question_types.items(), key=lambda x: x[1], reverse=True):
        percentage = count / 100 * 100
        print(f"  - {qtype.capitalize()}: {count} ({percentage:.1f}%)")
    
    # 测试特定功能组合
    print(f"\n{'='*60}")
    print("Testing SymPy special features:")
    
    # 测试三角形类型判断
    print("\nTesting triangle type detection:")
    triangle_types = [
        {"name": "Right triangle", "vertices": {"A": [0, 0], "B": [3, 0], "C": [0, 4]}},
        {"name": "Equilateral triangle", "vertices": {"A": [0, 0], "B": [1, 0], "C": [0.5, 0.866]}},
        {"name": "Isosceles triangle", "vertices": {"A": [0, 0], "B": [2, 0], "C": [1, 3]}}
    ]
    
    for tri in triangle_types:
        print(f"\n  {tri['name']}:")
        type_action = {
            "function": "triangle_type",
            "args": tri['vertices']
        }
        
        result = sympy_tool.execute(json.dumps(type_action))
        if result.get('success'):
            type_info = result.get('result', {})
            print(f"    - Is equilateral: {type_info.get('is_equilateral')}")
            print(f"    - Is isosceles: {type_info.get('is_isosceles')}")
            print(f"    - Is right: {type_info.get('is_right')}")
            print(f"    - Sides: {type_info.get('sides')}")
    
    # 测试点到直线距离
    print("\n\nTesting point to line distance:")
    distance_action = {
        "function": "distance_point_to_line",
        "args": {
            "point": [2, 3],
            "line_p1": [0, 0],
            "line_p2": [1, 1]
        }
    }
    
    result = sympy_tool.execute(json.dumps(distance_action))
    if result.get('success'):
        dist_result = result.get('result', {})
        print(f"  - Distance: {dist_result.get('distance')}")
        print(f"  - Foot of perpendicular: {dist_result.get('foot_of_perpendicular')}")
    
    # 保存分析结果
    analysis_result = {
        "total_samples": len(data),
        "analyzed_samples": 100,
        "question_types": question_types,
        "output_directory": output_dir
    }
    
    analysis_path = os.path.join(output_dir, "analysis_results.json")
    with open(analysis_path, 'w') as f:
        json.dump(analysis_result, f, indent=2)
    print(f"\nAnalysis results saved to: {analysis_path}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# 总结
print(f"\n{'='*60}")
print("Summary:")
print("1. SymPy几何工具可以处理各种几何计算")
print("2. 对于UniGeo数据集，主要功能：")
print("   - 三角形角度和面积计算")
print("   - 勾股定理应用")
print("   - 圆的相关计算")
print("   - 多边形面积和周长")
print("   - 线段夹角和几何关系判断")
print("\n注意事项：")
print("- 需要从问题文本中提取几何参数（坐标、长度等）")
print("- 可以结合图像识别工具获取准确的几何信息")
print("- SymPy提供精确的符号计算，适合几何证明")
print("\n建议：")
print("- 可以先用OCR或Grounding DINO提取图像中的几何信息")
print("- 然后用SymPy进行精确计算")
print("- 对于复杂问题，可以组合多个几何计算功能")
print("- 计算结果可以帮助VLM验证和理解几何关系")