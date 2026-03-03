# fix_reset_method.py
"""
手动修复 reset 方法中的 resize 功能
"""
import os
import shutil
from datetime import datetime

TOOL_PATH = '/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/vlm_gym/environments/tools/medsam2.py'
BACKUP_PATH = TOOL_PATH + f'.backup_fix_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

print("=== Fixing Reset Method ===\n")

# 1. 备份
print("1. Creating backup...")
shutil.copy2(TOOL_PATH, BACKUP_PATH)
print(f"   ✓ Backup saved to: {BACKUP_PATH}")

# 2. 读取文件
print("\n2. Reading file...")
with open(TOOL_PATH, 'r') as f:
    lines = f.readlines()

print(f"   Total lines: {len(lines)}")

# 3. 定位需要修改的行
print("\n3. Locating lines to modify...")
modifications = []

# 根据检查结果，我们需要修改第244、250、255行
target_lines = {
    244: {
        'old': 'self.current_image_np = np.array(self.current_image)',
        'new': [
            '# 调整图像尺寸以符合模型要求\n',
            'processed_array, orig_size, resize_info = self._prepare_image(self.current_image)\n',
            'self.current_image_np = processed_array\n',
            'self.resize_info = resize_info  # 保存resize信息\n'
        ]
    },
    250: {
        'old': 'self.current_image_np = np.array(self.current_image)',
        'new': [
            '# 调整图像尺寸以符合模型要求\n',
            'processed_array, orig_size, resize_info = self._prepare_image(self.current_image)\n',
            'self.current_image_np = processed_array\n',
            'self.resize_info = resize_info  # 保存resize信息\n'
        ]
    },
    255: {
        'old': 'self.current_frames = [np.array(img.convert("RGB")) for img in image]',
        'new': [
            '# 处理每一帧\n',
            'self.current_frames = []\n',
            'self.resize_info_list = []\n',
            'for img in image:\n',
            '    img_rgb = img.convert("RGB")\n',
            '    processed_array, orig_size, resize_info = self._prepare_image(img_rgb)\n',
            '    self.current_frames.append(processed_array)\n',
            '    self.resize_info_list.append(resize_info)\n',
            'self.resize_info = self.resize_info_list[len(image)//2]  # 中间帧的resize信息\n'
        ]
    }
}

# 4. 执行修改
print("\n4. Applying modifications...")
# 从后往前修改，避免行号变化的问题
for line_num in sorted(target_lines.keys(), reverse=True):
    line_idx = line_num - 1  # 转换为0-based索引
    
    if line_idx < len(lines):
        current_line = lines[line_idx]
        expected = target_lines[line_num]['old']
        
        # 验证是否是期望的行
        if expected in current_line:
            print(f"   Line {line_num}: Found expected pattern")
            
            # 获取缩进
            indent = len(current_line) - len(current_line.lstrip())
            indent_str = ' ' * indent
            
            # 准备新的行（保持相同缩进）
            new_lines = []
            for new_line in target_lines[line_num]['new']:
                # 如果新行不是空行，添加缩进
                if new_line.strip():
                    new_lines.append(indent_str + new_line)
                else:
                    new_lines.append(new_line)
            
            # 替换
            lines[line_idx:line_idx+1] = new_lines
            print(f"   ✓ Replaced with {len(new_lines)} lines")
        else:
            print(f"   ⚠️  Line {line_num}: Pattern not found")
            print(f"      Expected: {expected}")
            print(f"      Found: {current_line.strip()}")

# 5. 查找并修复 self.current_frames = [self.current_image_np] 的情况
print("\n5. Fixing additional patterns...")
i = 0
additional_fixes = 0
while i < len(lines):
    line = lines[i]
    
    # 查找需要添加 resize_info_list 的地方
    if 'self.current_frames = [self.current_image_np]' in line and i+1 < len(lines):
        next_line = lines[i+1]
        # 检查下一行是否已经有 resize_info_list
        if 'resize_info_list' not in next_line:
            indent = len(line) - len(line.lstrip())
            indent_str = ' ' * indent
            new_line = indent_str + 'self.resize_info_list = [self.resize_info]  # 保存为列表格式\n'
            lines.insert(i+1, new_line)
            additional_fixes += 1
            print(f"   Added resize_info_list after line {i+1}")
            i += 2
            continue
    i += 1

print(f"   ✓ Made {additional_fixes} additional fixes")

# 6. 写入文件
print("\n6. Writing modified file...")
with open(TOOL_PATH, 'w') as f:
    f.writelines(lines)
print("   ✓ File written")

# 7. 验证
print("\n7. Validating modifications...")
try:
    with open(TOOL_PATH, 'r') as f:
        content = f.read()
    
    # 编译检查
    compile(content, TOOL_PATH, 'exec')
    print("   ✓ File is valid Python")
    
    # 检查修改是否成功
    remaining_old_patterns = 0
    for target in target_lines.values():
        if target['old'] in content:
            # 检查是否在_prepare_image或_restore_original_size方法内
            # 这些方法内可能合法地包含这些模式
            if not ('def _prepare_image' in content and content.find('def _prepare_image') < content.find(target['old']) < content.find('def _restore_original_size')):
                remaining_old_patterns += 1
    
    if remaining_old_patterns == 0:
        print("   ✓ All old patterns successfully replaced")
    else:
        print(f"   ⚠️  {remaining_old_patterns} old patterns might still exist")
    
    # 检查新代码
    if 'self._prepare_image' in content:
        occurrences = content.count('self._prepare_image')
        print(f"   ✓ Found {occurrences} calls to _prepare_image")
    else:
        print("   ✗ No calls to _prepare_image found!")
        
except SyntaxError as e:
    print(f"   ✗ Syntax error: {e}")
    print("   Restoring backup...")
    shutil.copy2(BACKUP_PATH, TOOL_PATH)
    print("   ✓ Backup restored")
    exit(1)

print("\n=== Fix Complete ===")
print(f"✅ Reset method has been fixed!")
print(f"   Backup saved to: {BACKUP_PATH}")
print(f"\nNext step: Run test_medsam2_resize.py to verify the fix")