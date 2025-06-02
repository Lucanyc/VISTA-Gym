#!/usr/bin/env python
"""check structure of dataset"""
import os
import json
from pathlib import Path
import sys

def check_dataset_structure(data_root: str):

    data_root = Path(data_root)
    
    print(f"Checking dataset at: {data_root}")
    print("="*60)
    
    # check ChartQA
    chartqa_dir = data_root / "chartqa"
    if chartqa_dir.exists():
        print("\n[ChartQA Dataset]")
        check_directory(chartqa_dir)
    else:
        print("\n[!] ChartQA directory not found")
        
    # check ScienceQA  
    scienceqa_dir = data_root / "scienceqa"
    if scienceqa_dir.exists():
        print("\n[ScienceQA Dataset]")
        check_directory(scienceqa_dir)
    else:
        print("\n[!] ScienceQA directory not found")

def check_directory(dir_path: Path):


    file_types = {}
    total_size = 0
    
    for item in dir_path.rglob("*"):
        if item.is_file():
            ext = item.suffix.lower()
            file_types[ext] = file_types.get(ext, 0) + 1
            total_size += item.stat().st_size
            
    print(f"Directory: {dir_path}")
    print(f"Total files: {sum(file_types.values())}")
    print(f"Total size: {total_size / (1024*1024):.1f} MB")
    print(f"File types: {dict(file_types)}")
    

    json_files = list(dir_path.glob("*.json"))
    if json_files:
        print(f"\nFound {len(json_files)} JSON files:")
        for jf in json_files[:5]: 
            print(f"  - {jf.name}")
            try:
                with open(jf, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        print(f"    Type: list, Length: {len(data)}")
                        if data:
                            print(f"    First item keys: {list(data[0].keys())}")
                    elif isinstance(data, dict):
                        print(f"    Type: dict, Keys: {list(data.keys())[:5]}")
            except Exception as e:
                print(f"    Error reading: {e}")
                

    image_exts = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
    image_files = []
    for ext in image_exts:
        image_files.extend(list(dir_path.glob(f"*{ext}")))
        image_files.extend(list(dir_path.glob(f"*{ext.upper()}")))
        
    if image_files:
        print(f"\nFound {len(image_files)} image files")
        print(f"Sample images: {[f.name for f in image_files[:3]]}")

if __name__ == "__main__":
    data_path = "/data/wang/meng/GYM-Work/try_vlm_gym/data/llava_cot_images"
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    check_dataset_structure(data_path)
