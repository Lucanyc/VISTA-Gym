#!/usr/bin/env python3
"""
Convert AI2D dataset to VLM Gym format

This script converts the extracted AI2D dataset (JSON + images) to the VLM Gym format
that can be used for evaluation.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_answer_to_letter(answer_idx: str, choices: List[str]) -> str:
    """Convert answer index to answer letter and value"""
    try:
        idx = int(answer_idx)
        letter = chr(65 + idx)  # Convert 0->A, 1->B, etc.
        return letter
    except:
        return answer_idx


def get_answer_value(answer_idx: str, choices: List[str]) -> str:
    """Get the actual answer value from choices"""
    try:
        idx = int(answer_idx)
        if 0 <= idx < len(choices):
            return choices[idx]
        else:
            logger.warning(f"Answer index {idx} out of range for choices {choices}")
            return answer_idx
    except:
        return answer_idx


def process_ai2d_entry(
    item: Dict[str, Any],
    images_dir: Path,
    base_output_path: Path
) -> Optional[Dict[str, Any]]:
    """
    Convert a single AI2D entry to VLM Gym format
    
    Args:
        item: AI2D data entry
        images_dir: Directory containing extracted images
        base_output_path: Base path for output (for relative paths)
        
    Returns:
        VLM Gym format dictionary or None if error
    """
    
    # Get image path
    image_filename = item.get('image_filename', '')
    if not image_filename:
        logger.warning(f"No image filename for item {item.get('id', 'unknown')}")
        return None
    
    image_path = images_dir / image_filename
    if not image_path.exists():
        logger.warning(f"Image not found: {image_path}")
        return None
    
    # Get question and choices
    question = item.get('question', '')
    choices = item.get('options', [])
    answer_idx = item.get('answer', '')
    
    # Convert answer
    answer_letter = convert_answer_to_letter(answer_idx, choices)
    answer_value = get_answer_value(answer_idx, choices)
    
    # Create VLM Gym format entry
    vlm_entry = {
        "id": f"ai2d_test_{item.get('id', item.get('index', 0))}",
        "dataset": "ai2d",
        "image_path": str(image_path.absolute()),
        "question": question,
        "answer": answer_value,  # The actual answer text
        "choices": choices,
        "task": "diagram_qa",  # AI2D is primarily diagram understanding
        "metadata": {
            "original_id": item.get('id', item.get('index', 0)),
            "split": "test",
            "answer_letter": answer_letter,
            "answer_index": answer_idx,
            "image_filename": image_filename,
            "question_type": "multiple_choice",
            "num_choices": len(choices),
            "has_image": True,
            "language": "en",
            "domain": "science_diagram"
        }
    }
    
    # Add answer_text if available
    if 'answer_text' in item:
        vlm_entry["metadata"]["original_answer_text"] = item['answer_text']
    
    return vlm_entry


def convert_ai2d_dataset(
    json_file: Path,
    images_dir: Path,
    output_dir: Path,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Convert AI2D dataset to VLM Gym format
    
    Args:
        json_file: Path to the extracted AI2D JSON file
        images_dir: Directory containing extracted images
        output_dir: Output directory for VLM Gym JSON
        limit: Optional limit on number of samples
        
    Returns:
        List of converted entries
    """
    
    # Load AI2D data
    logger.info(f"Loading AI2D data from: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        ai2d_data = json.load(f)
    
    logger.info(f"Loaded {len(ai2d_data)} entries")
    
    # Apply limit if specified
    if limit:
        ai2d_data = ai2d_data[:limit]
        logger.info(f"Limited to {len(ai2d_data)} entries")
    
    # Convert each entry
    vlm_entries = []
    failed_count = 0
    
    for item in ai2d_data:
        vlm_entry = process_ai2d_entry(
            item=item,
            images_dir=images_dir,
            base_output_path=output_dir
        )
        
        if vlm_entry:
            vlm_entries.append(vlm_entry)
        else:
            failed_count += 1
    
    logger.info(f"Successfully converted {len(vlm_entries)} entries")
    if failed_count > 0:
        logger.warning(f"Failed to convert {failed_count} entries")
    
    # Save to output file
    output_file = output_dir / "ai2d_test_vlmgym.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(vlm_entries, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved VLM Gym format data to: {output_file}")
    
    return vlm_entries


def analyze_dataset(entries: List[Dict[str, Any]]) -> None:
    """Print statistics about the converted dataset"""
    
    print("\n" + "="*60)
    print("AI2D DATASET ANALYSIS")
    print("="*60)
    
    print(f"\nTotal samples: {len(entries)}")
    
    # Question length statistics
    question_lengths = [len(entry['question'].split()) for entry in entries]
    avg_question_length = sum(question_lengths) / len(question_lengths) if question_lengths else 0
    
    print(f"\nQuestion statistics:")
    print(f"  Average length: {avg_question_length:.1f} words")
    print(f"  Min length: {min(question_lengths)} words")
    print(f"  Max length: {max(question_lengths)} words")
    
    # Choices statistics
    num_choices = [len(entry['choices']) for entry in entries]
    choices_distribution = {}
    for n in num_choices:
        choices_distribution[n] = choices_distribution.get(n, 0) + 1
    
    print(f"\nNumber of choices distribution:")
    for n_choices, count in sorted(choices_distribution.items()):
        print(f"  {n_choices} choices: {count} questions ({count/len(entries)*100:.1f}%)")
    
    # Answer distribution
    answer_letters = [entry['metadata']['answer_letter'] for entry in entries]
    answer_distribution = {}
    for letter in answer_letters:
        answer_distribution[letter] = answer_distribution.get(letter, 0) + 1
    
    print(f"\nAnswer distribution:")
    for letter, count in sorted(answer_distribution.items()):
        print(f"  {letter}: {count} ({count/len(entries)*100:.1f}%)")


def verify_data_integrity(entries: List[Dict[str, Any]], sample_size: int = 5) -> None:
    """Verify data integrity and show sample entries"""
    
    # Check for missing images
    missing_images = []
    for entry in entries:
        if not Path(entry['image_path']).exists():
            missing_images.append(entry['id'])
    
    if missing_images:
        print(f"\n⚠️  Warning: {len(missing_images)} images not found")
        print(f"First few missing: {missing_images[:5]}")
    else:
        print(f"\n✓ All {len(entries)} images verified")
    
    # Show sample entries
    print(f"\n" + "="*60)
    print(f"SAMPLE ENTRIES (first {sample_size})")
    print("="*60)
    
    for i, entry in enumerate(entries[:sample_size]):
        print(f"\nEntry {i+1}:")
        print(f"  ID: {entry['id']}")
        print(f"  Question: {entry['question']}")
        print(f"  Choices: {entry['choices']}")
        print(f"  Answer: {entry['answer']} (Letter: {entry['metadata']['answer_letter']})")
        print(f"  Image: {Path(entry['image_path']).name}")


def create_split_files(
    entries: List[Dict[str, Any]], 
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> None:
    """
    Create train/val/test splits from the data
    
    Args:
        entries: All data entries
        output_dir: Output directory
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
    """
    import random
    
    # Shuffle entries
    shuffled_entries = entries.copy()
    random.seed(42)  # For reproducibility
    random.shuffle(shuffled_entries)
    
    # Calculate split sizes
    total = len(shuffled_entries)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    # Split data
    train_data = shuffled_entries[:train_size]
    val_data = shuffled_entries[train_size:train_size + val_size]
    test_data = shuffled_entries[train_size + val_size:]
    
    # Update IDs and metadata
    for entry in train_data:
        entry['id'] = entry['id'].replace('_test_', '_train_')
        entry['metadata']['split'] = 'train'
    
    for entry in val_data:
        entry['id'] = entry['id'].replace('_test_', '_val_')
        entry['metadata']['split'] = 'val'
    
    # test_data already has 'test' split
    
    # Save splits
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    for split_name, split_data in splits.items():
        output_file = output_dir / f"ai2d_{split_name}_vlmgym.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {split_name} split: {len(split_data)} samples to {output_file}")
    
    print(f"\nSplit statistics:")
    print(f"  Train: {len(train_data)} samples ({len(train_data)/total*100:.1f}%)")
    print(f"  Val: {len(val_data)} samples ({len(val_data)/total*100:.1f}%)")
    print(f"  Test: {len(test_data)} samples ({len(test_data)/total*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert AI2D dataset to VLM Gym format"
    )
    parser.add_argument(
        "--ai2d-root",
        type=str,
        default=None,
        help="Root directory of AI2D dataset"
    )
    parser.add_argument(
        "--extracted-dir",
        type=str,
        default=None,
        help="Directory with extracted data (default: ai2d-root/extracted)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for VLM Gym format files"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to convert (for testing)"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze the dataset after conversion"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify data integrity and show samples"
    )
    parser.add_argument(
        "--create-splits",
        action="store_true",
        help="Create train/val/test splits"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    ai2d_root = Path(args.ai2d_root)
    
    if args.extracted_dir:
        extracted_dir = Path(args.extracted_dir)
    else:
        extracted_dir = ai2d_root / "extracted"
    
    json_file = extracted_dir / "ai2d_test_all.json"
    images_dir = extracted_dir / "images"
    output_dir = Path(args.output_dir)
    
    # Validate paths
    if not json_file.exists():
        logger.error(f"JSON file not found: {json_file}")
        return
    
    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        return
    
    # Convert dataset
    logger.info("Starting AI2D to VLM Gym conversion...")
    entries = convert_ai2d_dataset(
        json_file=json_file,
        images_dir=images_dir,
        output_dir=output_dir,
        limit=args.limit
    )
    
    # Analyze if requested
    if args.analyze:
        analyze_dataset(entries)
    
    # Verify if requested
    if args.verify:
        verify_data_integrity(entries)
    
    # Create splits if requested
    if args.create_splits:
        create_split_files(entries, output_dir)
    
    print(f"\n✓ Conversion complete!")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
