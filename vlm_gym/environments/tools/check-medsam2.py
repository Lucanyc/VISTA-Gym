# check_current_medsam2_tool.py
"""
检查当前的MedSAM2工具实现，看是否有resize功能
"""
import os

print("=== Checking Current MedSAM2 Tool Implementation ===\n")

# 检查medsam2.py文件
tool_path = '/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/vlm_gym/environments/tools/medsam2.py'

if os.path.exists(tool_path):
    with open(tool_path, 'r') as f:
        content = f.read()
    
    print("1. Checking for resize-related code...")
    
    # 查找resize相关的代码
    resize_keywords = ['resize', 'transform', 'scale', 'interpolate', '512']
    found_keywords = []
    
    lines = content.split('\n')
    for i, line in enumerate(lines):
        for keyword in resize_keywords:
            if keyword.lower() in line.lower():
                found_keywords.append((i+1, line.strip()))
    
    if found_keywords:
        print("Found resize-related code:")
        for line_num, line in found_keywords:
            print(f"  Line {line_num}: {line}")
    else:
        print("  No resize-related code found")
    
    # 检查reset方法
    print("\n2. Checking reset method...")
    reset_start = content.find('def reset(')
    if reset_start != -1:
        reset_end = content.find('\n    def ', reset_start + 1)
        if reset_end == -1:
            reset_end = len(content)
        
        reset_method = content[reset_start:reset_end]
        print("Reset method found:")
        
        # 检查是否直接使用原始图像
        if 'np.array(image)' in reset_method:
            print("  - Converts image to numpy array")
        if 'resize' not in reset_method.lower():
            print("  - ⚠️  NO RESIZE operation found in reset method")
        else:
            print("  - ✓ Resize operation found")
    
    # 检查_prepare_image方法（如果存在）
    print("\n3. Checking for image preparation methods...")
    if '_prepare_image' in content:
        print("  - Found _prepare_image method")
    else:
        print("  - No _prepare_image method found")
    
    # 查找图像处理相关的导入
    print("\n4. Checking imports...")
    import_lines = [line for line in lines if line.strip().startswith('import') or line.strip().startswith('from')]
    
    image_processing_imports = []
    for line in import_lines:
        if any(keyword in line for keyword in ['PIL', 'Image', 'cv2', 'transform', 'resize']):
            image_processing_imports.append(line.strip())
    
    if image_processing_imports:
        print("Image processing imports found:")
        for imp in image_processing_imports:
            print(f"  - {imp}")
    else:
        print("  - No image processing imports found")
    
    # 提取reset方法的关键部分
    print("\n5. Key parts of reset method:")
    if reset_start != -1:
        # 找到处理图像的部分
        reset_lines = reset_method.split('\n')
        for i, line in enumerate(reset_lines):
            if 'video =' in line or 'images =' in line or 'np.array' in line:
                # 打印这行和前后几行
                start = max(0, i-2)
                end = min(len(reset_lines), i+3)
                print(f"\n  Around image processing (lines {start}-{end}):")
                for j in range(start, end):
                    prefix = ">>>" if j == i else "   "
                    print(f"  {prefix} {reset_lines[j]}")

else:
    print(f"Error: Tool file not found at {tool_path}")

print("\n=== Analysis Complete ===")
print("\nBased on the check, we need to determine if the tool:")
print("1. Already handles resizing (unlikely based on errors)")
print("2. Needs resize functionality added")
print("3. Has resize but it's not working correctly")