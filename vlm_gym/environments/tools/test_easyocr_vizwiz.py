#!/usr/bin/env python3
"""
Enhanced EasyOCR Testing on VizWiz Dataset
EasyOCR as an auxiliary tool for text-reading questions
"""

import sys
import json
import os
from pathlib import Path
import time
from collections import defaultdict, Counter
import re
import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    import easyocr
    from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
    import cv2
    print("All modules imported successfully")
    
    # 创建reader（支持英文）
    print("Creating EasyOCR reader...")
    reader = easyocr.Reader(['en'], gpu=True)
    print("Reader created successfully")
    
except Exception as e:
    print(f"Import error: {e}")
    sys.exit(1)


class VizWizOCRHelper:
    """Helper class for using OCR with VizWiz dataset"""
    
    def __init__(self, reader):
        self.reader = reader
        self.text_keywords = [
            'read', 'say', 'says', 'text', 'label', 'write', 'written',
            'name', 'brand', 'title', 'expiration', 'date', 'ingredient',
            'instruction', 'direction', 'price', 'number', 'word',
            'letter', 'sign', 'package', 'bottle', 'box', 'can'
        ]
        
    def is_text_question(self, question: str) -> bool:
        """Check if the question is text-related"""
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in self.text_keywords)
    
    def get_question_type(self, question: str) -> str:
        """Categorize the text question type"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['expiration', 'expire', 'date', 'best by']):
            return 'expiration_date'
        elif any(word in question_lower for word in ['ingredient', 'contain', 'sodium', 'sugar']):
            return 'ingredients'
        elif any(word in question_lower for word in ['name', 'brand', 'product', 'called']):
            return 'product_name'
        elif any(word in question_lower for word in ['instruction', 'direction', 'how to']):
            return 'instructions'
        elif any(word in question_lower for word in ['price', 'cost', '$']):
            return 'price'
        elif 'can' in question_lower or 'bottle' in question_lower:
            return 'container_label'
        else:
            return 'general_text'
    
    def preprocess_image(self, image: Image.Image, question_type: str) -> Image.Image:
        """Preprocess image based on question type"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Base preprocessing
        # 1. Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        # 2. Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.5)
        
        # Question-type specific preprocessing
        if question_type in ['expiration_date', 'ingredients', 'price']:
            # These often have small text, need more enhancement
            image = image.filter(ImageFilter.SHARPEN)
            
        elif question_type == 'container_label':
            # Might be curved surface, try to enhance edges
            image = image.filter(ImageFilter.EDGE_ENHANCE)
            
        return image
    
    def extract_text_multi_strategy(self, image_path: str, question: str) -> Dict:
        """Extract text using multiple strategies"""
        results = {
            'success': False,
            'method': None,
            'detected_texts': [],
            'all_text': '',
            'confidence': 0.0,
            'processing_time': 0.0
        }
        
        start_time = time.time()
        question_type = self.get_question_type(question)
        
        try:
            # Strategy 1: Direct OCR
            print("  Strategy 1: Direct OCR...")
            direct_result = self.reader.readtext(image_path, detail=1)
            
            if direct_result and len(direct_result) > 0:
                results['detected_texts'] = [(text, conf) for _, text, conf in direct_result]
                results['all_text'] = ' '.join([text for _, text, _ in direct_result])
                results['confidence'] = np.mean([conf for _, _, conf in direct_result])
                results['method'] = 'direct'
                results['success'] = True
                
                # If confidence is high enough, return
                if results['confidence'] > 0.7:
                    results['processing_time'] = time.time() - start_time
                    return results
            
            # Strategy 2: Preprocessed OCR
            print("  Strategy 2: Preprocessed OCR...")
            image = Image.open(image_path)
            preprocessed = self.preprocess_image(image, question_type)
            
            # Convert to numpy array for OCR
            preprocessed_np = np.array(preprocessed)
            enhanced_result = self.reader.readtext(preprocessed_np, detail=1)
            
            if enhanced_result and len(enhanced_result) > 0:
                enhanced_conf = np.mean([conf for _, _, conf in enhanced_result])
                
                # Use enhanced result if better
                if enhanced_conf > results['confidence']:
                    results['detected_texts'] = [(text, conf) for _, text, conf in enhanced_result]
                    results['all_text'] = ' '.join([text for _, text, _ in enhanced_result])
                    results['confidence'] = enhanced_conf
                    results['method'] = 'preprocessed'
                    results['success'] = True
            
            # Strategy 3: Region-based OCR for specific types
            if question_type in ['expiration_date', 'ingredients'] and results['confidence'] < 0.5:
                print("  Strategy 3: Region-based OCR...")
                # Try to focus on specific regions (bottom for ingredients, top for product name)
                height, width = image.size[1], image.size[0]
                
                if question_type == 'ingredients':
                    # Focus on bottom half
                    cropped = image.crop((0, height//2, width, height))
                else:
                    # Focus on top half
                    cropped = image.crop((0, 0, width, height//2))
                
                cropped_np = np.array(cropped)
                region_result = self.reader.readtext(cropped_np, detail=1)
                
                if region_result:
                    region_conf = np.mean([conf for _, _, conf in region_result])
                    if region_conf > results['confidence']:
                        results['detected_texts'] = [(text, conf) for _, text, conf in region_result]
                        results['all_text'] = ' '.join([text for _, text, _ in region_result])
                        results['confidence'] = region_conf
                        results['method'] = 'region_based'
                        results['success'] = True
            
        except Exception as e:
            print(f"  OCR error: {e}")
            results['success'] = False
            
        results['processing_time'] = time.time() - start_time
        return results
    
    def match_answer(self, ocr_results: Dict, expected_answer: str, all_answers: List[str]) -> Dict:
        """Match OCR results with expected answers"""
        match_result = {
            'exact_match': False,
            'partial_match': False,
            'annotator_match': False,
            'match_score': 0.0,
            'matched_text': None
        }
        
        if not ocr_results['success']:
            return match_result
        
        ocr_text_lower = ocr_results['all_text'].lower()
        expected_lower = expected_answer.lower()
        
        # Exact match
        if expected_lower in ocr_text_lower:
            match_result['exact_match'] = True
            match_result['match_score'] = 1.0
            match_result['matched_text'] = expected_answer
            return match_result
        
        # Check all annotator answers
        for annotator_answer in all_answers:
            if annotator_answer.lower() in ocr_text_lower:
                match_result['annotator_match'] = True
                match_result['match_score'] = 0.9
                match_result['matched_text'] = annotator_answer
                return match_result
        
        # Partial match - check key words
        expected_words = expected_lower.split()
        matched_words = 0
        for word in expected_words:
            if len(word) > 3 and word in ocr_text_lower:
                matched_words += 1
        
        if matched_words > 0:
            match_result['partial_match'] = True
            match_result['match_score'] = matched_words / len(expected_words)
            match_result['matched_text'] = f"{matched_words}/{len(expected_words)} words matched"
        
        return match_result
    
    def generate_ocr_confidence_score(self, ocr_results: Dict, match_result: Dict, 
                                    question_confidence: float) -> float:
        """Generate overall confidence score for OCR results"""
        if not ocr_results['success']:
            return 0.0
        
        # Base OCR confidence
        ocr_conf = ocr_results['confidence']
        
        # Match quality
        match_score = match_result['match_score']
        
        # Question confidence (from annotators)
        q_conf = question_confidence
        
        # Weighted average
        # Higher weight on match score for high-confidence questions
        if q_conf > 0.7:
            final_score = 0.3 * ocr_conf + 0.5 * match_score + 0.2 * q_conf
        else:
            # For low-confidence questions, rely more on OCR confidence
            final_score = 0.5 * ocr_conf + 0.3 * match_score + 0.2 * q_conf
        
        return final_score


def test_enhanced_ocr_pipeline():
    """Test the enhanced OCR pipeline on VizWiz"""
    
    # Initialize helper
    helper = VizWizOCRHelper(reader)
    
    # Load VizWiz data
    data_path = "/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/VizWiz/vizwiz_train_vlmgym.json"
    
    print(f"\nLoading VizWiz data from: {data_path}")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    # Filter text-related questions
    text_questions = [s for s in data if helper.is_text_question(s['question'])]
    print(f"Text-related questions: {len(text_questions)} ({len(text_questions)/len(data)*100:.1f}%)")
    
    # Categorize by question type
    question_type_dist = defaultdict(int)
    for q in text_questions[:100]:
        q_type = helper.get_question_type(q['question'])
        question_type_dist[q_type] += 1
    
    print("\nText question type distribution (sample of 100):")
    for q_type, count in sorted(question_type_dist.items(), key=lambda x: -x[1]):
        print(f"  {q_type}: {count}")
    
    # Test samples selection
    test_samples = []
    
    # Get diverse samples
    for conf_range, conf_name in [
        ((0.8, 1.0), 'high'),
        ((0.5, 0.8), 'medium'),
        ((0.0, 0.5), 'low')
    ]:
        range_samples = [
            s for s in text_questions 
            if conf_range[0] <= s['metadata']['answer_confidence'] < conf_range[1]
        ][:3]
        test_samples.extend(range_samples)
    
    print(f"\nTesting {len(test_samples)} samples with enhanced pipeline")
    print("="*80)
    
    # Results tracking
    results_by_confidence = defaultdict(lambda: {'total': 0, 'success': 0, 'partial': 0})
    results_by_method = defaultdict(int)
    results_by_question_type = defaultdict(lambda: {'total': 0, 'success': 0})
    
    # Test each sample
    for idx, sample in enumerate(test_samples):
        print(f"\nSample {idx + 1}/{len(test_samples)}")
        print("-"*60)
        
        question = sample['question']
        expected_answer = sample['answer']
        all_answers = sample['metadata']['all_answers']
        answer_confidence = sample['metadata']['answer_confidence']
        image_path = sample['image_path']
        
        print(f"Question: {question}")
        print(f"Expected: {expected_answer}")
        print(f"Confidence: {answer_confidence:.3f}")
        
        # Determine confidence level
        if answer_confidence >= 0.8:
            conf_level = 'high'
        elif answer_confidence >= 0.5:
            conf_level = 'medium'
        else:
            conf_level = 'low'
        
        results_by_confidence[conf_level]['total'] += 1
        
        # Get question type
        q_type = helper.get_question_type(question)
        results_by_question_type[q_type]['total'] += 1
        
        if os.path.exists(image_path):
            # Run OCR with multiple strategies
            ocr_results = helper.extract_text_multi_strategy(image_path, question)
            
            if ocr_results['success']:
                print(f"\nOCR Success! Method: {ocr_results['method']}")
                print(f"OCR Confidence: {ocr_results['confidence']:.3f}")
                print(f"Processing time: {ocr_results['processing_time']:.2f}s")
                print(f"Detected {len(ocr_results['detected_texts'])} text regions")
                
                # Show high-confidence texts
                high_conf_texts = [(t, c) for t, c in ocr_results['detected_texts'] if c > 0.5]
                if high_conf_texts:
                    print("\nHigh-confidence texts:")
                    for text, conf in high_conf_texts[:5]:
                        print(f"  '{text}' ({conf:.3f})")
                
                # Match with answers
                match_result = helper.match_answer(ocr_results, expected_answer, all_answers)
                
                # Calculate overall confidence
                overall_confidence = helper.generate_ocr_confidence_score(
                    ocr_results, match_result, answer_confidence
                )
                
                print(f"\nMatching Results:")
                print(f"  Exact match: {match_result['exact_match']}")
                print(f"  Annotator match: {match_result['annotator_match']}")
                print(f"  Partial match: {match_result['partial_match']}")
                print(f"  Match score: {match_result['match_score']:.3f}")
                print(f"  Overall confidence: {overall_confidence:.3f}")
                
                # Update statistics
                results_by_method[ocr_results['method']] += 1
                
                if match_result['exact_match'] or match_result['annotator_match']:
                    results_by_confidence[conf_level]['success'] += 1
                    results_by_question_type[q_type]['success'] += 1
                    print("  ✓ SUCCESS")
                elif match_result['partial_match']:
                    results_by_confidence[conf_level]['partial'] += 1
                    print("  ~ PARTIAL")
                else:
                    print("  ✗ NO MATCH")
                    
                # Decision logic
                if overall_confidence > 0.7:
                    print("\n→ Recommendation: Use OCR result with high confidence")
                elif overall_confidence > 0.4:
                    print("\n→ Recommendation: Use OCR as hint for VLM")
                else:
                    print("\n→ Recommendation: Rely on VLM only")
                    
            else:
                print("\nOCR Failed - no text detected")
                if expected_answer.lower() == 'unanswerable':
                    print("  ✓ Correctly identified as unanswerable")
                else:
                    print("  ✗ Failed to detect expected text")
        else:
            print(f"Image not found: {image_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("ENHANCED OCR PIPELINE SUMMARY")
    print("="*80)
    
    print("\nResults by Confidence Level:")
    for conf_level, stats in results_by_confidence.items():
        if stats['total'] > 0:
            success_rate = stats['success'] / stats['total'] * 100
            partial_rate = stats['partial'] / stats['total'] * 100
            print(f"\n{conf_level.upper()} confidence ({stats['total']} samples):")
            print(f"  Success rate: {success_rate:.1f}%")
            print(f"  Partial rate: {partial_rate:.1f}%")
            print(f"  Total useful: {success_rate + partial_rate:.1f}%")
    
    print("\nResults by OCR Method:")
    total_ocr = sum(results_by_method.values())
    for method, count in sorted(results_by_method.items(), key=lambda x: -x[1]):
        print(f"  {method}: {count} ({count/total_ocr*100:.1f}%)")
    
    print("\nResults by Question Type:")
    for q_type, stats in sorted(results_by_question_type.items(), key=lambda x: -x[1]['total']):
        if stats['total'] > 0:
            success_rate = stats['success'] / stats['total'] * 100
            print(f"  {q_type}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR VIZWIZ + OCR")
    print("="*80)
    
    print("\n1. OCR Confidence Thresholds:")
    print("   - > 0.7: Use OCR result directly")
    print("   - 0.4-0.7: Use as hint/context for VLM")
    print("   - < 0.4: Rely on VLM only")
    
    print("\n2. Best Use Cases:")
    print("   - Product names and brands")
    print("   - Clear printed labels")
    print("   - High-confidence text questions")
    
    print("\n3. Integration Strategy:")
    print("   - Pre-filter questions by type")
    print("   - Apply appropriate preprocessing")
    print("   - Use multi-strategy approach")
    print("   - Combine with VLM for robustness")


if __name__ == "__main__":
    test_enhanced_ocr_pipeline()