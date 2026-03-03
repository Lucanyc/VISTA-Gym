#!/usr/bin/env python3
"""Convert the standard ChartQA dataset to the unified VLM Gym format.

This script reads a ChartQA annotation JSON + its image directory and produces
a VLM Gym–compatible dataset, enabling evaluation and training under a unified
multimodal benchmark framework.
"""

# This script converts the downloaded ChartQA dataset into VLM Gym format.

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def convert_chartqa_to_vlmgym(
    chartqa_file: Path,
    image_dir: Path,
    output_file: Path,
    dataset_name: str = "chartqa"
) -> List[Dict[str, Any]]:
    """
    Convert ChartQA files into VLM Gym unified format.

    Expected ChartQA input format example:
        {
            "imgname": "two_col_1234.png",
            "query": "What is the highest value?",
            "label": "85"
        }

    Target VLM Gym unified format:
        {
            "id": "chartqa_0",
            "dataset": "chartqa",
            "image_path": "/path/to/image.png",
            "question": "What is the highest value?",
            "answer": "85",
            "choices": null,   # ChartQA does not provide multiple-choice selections
            "task": "chart_qa",
            "metadata": {
                "original_index": 0,
                "source_file": "...",
                "image_filename": "two_col_1234.png"
            }
        }

    Notes:
    - ChartQA contains open-ended chart reasoning questions without answer options.
    - This function preserves the raw answer field as-is and assigns `choices = None`.
    - The `id` field is constructed sequentially to maintain unique sample IDs.
    """
    
    # Load the original ChartQA annotation JSON into memory
    with open(chartqa_file, 'r') as f:
        chartqa_data = json.load(f)
    
    print(f"Loaded {len(chartqa_data)} samples from {chartqa_file}")
    
    # This will store converted VLM Gym–formatted items
    vlm_gym_data = []
    
    for idx, item in enumerate(chartqa_data):
        # Retrieve the raw annotation fields
        image_name = item.get('imgname', '')
        question = item.get('query', '')
        answer = item.get('label', '')
        
        # Construct the absolute image path using the image directory provided
        image_path = image_dir / image_name
        
        # Construct a new entry following the VLM Gym schema
        vlm_entry = {
            "id": f"{dataset_name}_{idx}",            # Unique ID for each sample
            "dataset": dataset_name,                 # Dataset name label
            "image_path": str(image_path),           # Path to the chart image
            "question": question,                    # Natural language question
            "answer": answer,                        # Ground-truth numeric/text answer
            "choices": None,                         # ChartQA is not multiple-choice
            "task": "chart_qa",                      # Task category for VLM Gym
            "metadata": {
                "original_index": idx,               # Original index in source file
                "source_file": chartqa_file.name,    # Which JSON file this sample came from
                "image_filename": image_name         # The raw image filename
            }
        }
        
        vlm_gym_data.append(vlm_entry)
    
    # Save all converted entries to the requested output JSON file
    with open(output_file, 'w') as f:
        json.dump(vlm_gym_data, f, indent=2)
    
    print(f"Converted {len(vlm_gym_data)} samples")
    print(f"Saved to: {output_file}")
    
    return vlm_gym_data


def verify_images(vlm_gym_data: List[Dict[str, Any]]) -> None:
    """
    Verify whether all image paths referenced in the VLM Gym dataset exist.

    This is a useful sanity check before running downstream VLM evaluations,
    as missing images will cause model inference to fail.
    """
    missing_images = []
    
    for item in vlm_gym_data:
        image_path = Path(item['image_path'])
        if not image_path.exists():
            missing_images.append(image_path)
    
    if missing_images:
        print(f"\nWarning: {len(missing_images)} images not found:")
        for img in missing_images[:5]:
            print(f"  - {img}")
        if len(missing_images) > 5:
            print(f"  ... and {len(missing_images) - 5} more")
    else:
        print(f"\n✓ All {len(vlm_gym_data)} images verified")


def merge_with_existing(
    new_data_file: Path,
    existing_file: Path,
    output_file: Path
) -> None:
    """
    Merge a newly generated ChartQA VLM Gym dataset with an existing VLM Gym dataset.

    The function:
    - Loads the existing unified dataset.
    - Removes any previous ChartQA entries from it.
    - Appends the newly converted ChartQA dataset.
    - Saves the merged version into a new output JSON.

    This is useful when incrementally adding ChartQA to a combined benchmark dataset.
    """
    
    # Load datasets
    with open(existing_file, 'r') as f:
        existing_data = json.load(f)
    with open(new_data_file, 'r') as f:
        new_data = json.load(f)
    
    print(f"Existing data: {len(existing_data)} samples")
    print(f"New ChartQA data: {len(new_data)} samples")
    
    # Remove any outdated ChartQA entries to avoid duplication
    filtered_existing = [
        item for item in existing_data
        if item.get('dataset', '').lower() != 'chartqa'
    ]
    
    print(f"After removing old ChartQA: {len(filtered_existing)} samples")
    
    # Combine preserved existing samples + new ChartQA samples
    merged_data = filtered_existing + new_data
    
    # Save merged combined dataset
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"Merged data: {len(merged_data)} samples")
    print(f"Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ChartQA dataset to VLM Gym format"
    )
    parser.add_argument(
        "--chartqa-file", 
        type=str, 
        required=True,
        help="Path to ChartQA JSON file (e.g., test_human.json)"
    )
    parser.add_argument(
        "--image-dir", 
        type=str, 
        required=True,
        help="Path to ChartQA image directory (folder containing .png charts)"
    )
    parser.add_argument(
        "--output-file", 
        type=str, 
        required=True,
        help="Output JSON file for writing the VLM Gym–formatted dataset"
    )
    parser.add_argument(
        "--dataset-name", 
        type=str, 
        default="chartqa",
        help="Dataset name tag to write into the converted entries"
    )
    parser.add_argument(
        "--merge-with", 
        type=str, 
        default=None,
        help="Optional: merge the converted dataset with an existing VLM Gym dataset JSON"
    )
    parser.add_argument(
        "--verify", 
        action="store_true",
        help="After conversion, verify that all image paths resolve correctly"
    )
    
    args = parser.parse_args()
    
    # Convert provided CLI arguments into path objects
    chartqa_file = Path(args.chartqa_file)
    image_dir = Path(args.image_dir)
    output_file = Path(args.output_file)
    
    # Ensure the required inputs are valid
    if not chartqa_file.exists():
        print(f"Error: ChartQA file not found: {chartqa_file}")
        return
    if not image_dir.exists():
        print(f"Error: Image directory not found: {image_dir}")
        return
    
    # Create output directory if missing
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Perform primary ChartQA → VLM Gym conversion
    vlm_gym_data = convert_chartqa_to_vlmgym(
        chartqa_file=chartqa_file,
        image_dir=image_dir,
        output_file=output_file,
        dataset_name=args.dataset_name
    )
    
    # Optionally perform image existence verification
    if args.verify:
        verify_images(vlm_gym_data)
    
    # Optionally merge with an existing unified dataset
    if args.merge_with:
        merge_file = Path(args.merge_with)
        if not merge_file.exists():
            print(f"Error: Merge file not found: {merge_file}")
            return
        
        merged_output = output_file.parent / f"merged_{output_file.name}"
        merge_with_existing(
            new_data_file=output_file,
            existing_file=merge_file,
            output_file=merged_output
        )


if __name__ == "__main__":
    main()
