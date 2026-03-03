#!/usr/bin/env python3
"""ChartQA-specific adapter for VLM Gym"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict
import random

logger = logging.getLogger(__name__)


class ChartQAAdapter:
    """
    Adapter specifically designed for ChartQA dataset
    
    Features:
    - Efficient data loading and caching
    - Multiple annotation file support
    - Task filtering by question type, difficulty, etc.
    - Data validation and preprocessing
    - Statistics and analysis
    """
    
    def __init__(
        self,
        data_root: str,
        annotation_files: Union[str, List[str]],
        cache_size: int = 1000,
        validate_images: bool = True
    ):
        """
        Initialize ChartQA adapter
        
        Args:
            data_root: Root directory containing images
            annotation_files: Path(s) to annotation JSON files
            cache_size: Maximum number of tasks to cache in memory
            validate_images: Whether to validate image paths on load
        """
        self.data_root = Path(data_root)
        self.cache_size = cache_size
        self.validate_images = validate_images
        
        # Ensure annotation_files is a list
        if isinstance(annotation_files, str):
            annotation_files = [annotation_files]
        self.annotation_files = [Path(f) for f in annotation_files]
        
        # Data storage
        self.annotations = []
        self._task_index = {}  # task_id -> annotation
        self._cache = {}  # LRU cache for task data
        self._cache_order = []  # Track cache order for LRU
        
        # Metadata
        self.stats = defaultdict(int)
        self.question_types = defaultdict(list)  # question_type -> task_ids
        self.image_index = defaultdict(list)  # image_path -> task_ids
        
        # Load annotations
        self._load_annotations()
    
    
    def _load_annotations(self):
        """Load and index all annotation files"""
        logger.info(f"Loading ChartQA annotations from {len(self.annotation_files)} files")
        
        for ann_file in self.annotation_files:
            if not ann_file.exists():
                logger.warning(f"Annotation file not found: {ann_file}")
                continue
                
            try:
                with open(ann_file, 'r') as f:
                    data = json.load(f)
                
                # Process each annotation
                for item in data:
                    # Ensure it's a ChartQA task
                    dataset_lower = item.get('dataset', '').lower()
                    task_lower = item.get('task', '').lower()
                    if not (dataset_lower.startswith('chartqa') or task_lower in ['chart_qa', 'chartqa']):
                        continue
                    
                    # Validate and fix paths
                    if self.validate_images:
                        item = self._validate_item(item)
                        if item is None:
                            continue
                    
                    # Add to collections
                    task_id = item['id']
                    self.annotations.append(item)
                    self._task_index[task_id] = item
                    
                    # Index by question type
                    q_type = self._classify_question(item['question'])
                    self.question_types[q_type].append(task_id)
                    
                    # Index by image
                    self.image_index[item['image_path']].append(task_id)
                    
                    # Update stats
                    self.stats['total'] += 1
                    self.stats[f'source_{ann_file.stem}'] += 1
                    self.stats[f'qtype_{q_type}'] += 1
                    
                logger.info(f"Loaded {self.stats[f'source_{ann_file.stem}']} tasks from {ann_file.name}")
                
            except Exception as e:
                logger.error(f"Error loading {ann_file}: {e}")
        
        logger.info(f"Total ChartQA tasks loaded: {self.stats['total']}")
        self._log_statistics()
    
    
    #def _load_annotations(self):
    #    """Load and index all annotation files"""
    #    logger.info(f"Loading ChartQA annotations from {len(self.annotation_files)} files")
        
    #    for ann_file in self.annotation_files:
    #        if not ann_file.exists():
    #            logger.warning(f"Annotation file not found: {ann_file}")
    #            continue
                
    #        try:
    #            with open(ann_file, 'r') as f:
    #                data = json.load(f)
                
                # Process each annotation
    #            for item in data:
                    # Ensure it's a ChartQA task
    #                if item.get('dataset', '').lower() != 'chartqa' and \
    #                   item.get('task', '').lower() not in ['chart_qa', 'chartqa']:
    #                    continue
                    
                    # Validate and fix paths
    #                if self.validate_images:
    #                    item = self._validate_item(item)
    #                    if item is None:
    #                        continue
                    
                    # Add to collections
    #                task_id = item['id']
    #                self.annotations.append(item)
    #                self._task_index[task_id] = item
                    
                    # Index by question type
    #                q_type = self._classify_question(item['question'])
    #                self.question_types[q_type].append(task_id)
                    
                    # Index by image
    #                self.image_index[item['image_path']].append(task_id)
                    
                    # Update stats
    #                self.stats['total'] += 1
    #                self.stats[f'source_{ann_file.stem}'] += 1
    #                self.stats[f'qtype_{q_type}'] += 1
                    
    #            logger.info(f"Loaded {self.stats[f'source_{ann_file.stem}']} tasks from {ann_file.name}")
                
    #        except Exception as e:
    #            logger.error(f"Error loading {ann_file}: {e}")
        
    #    logger.info(f"Total ChartQA tasks loaded: {self.stats['total']}")
    #    self._log_statistics()
    
    def _validate_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate and fix a single annotation item"""
        # Check required fields
        required = ['id', 'image_path', 'question', 'answer']
        for field in required:
            if field not in item:
                logger.warning(f"Missing required field '{field}' in task {item.get('id', 'unknown')}")
                return None
        
        # Fix image path
        image_path = Path(item['image_path'])
        
        # Try different path resolutions
        possible_paths = [
            self.data_root / image_path,  # Relative to data_root
            image_path,  # As is (might be absolute)
            self.data_root / image_path.name,  # Just filename in data_root
        ]
        
        # Find the correct path
        for path in possible_paths:
            if path.exists():
                item['image_path'] = str(path.resolve())
                return item
        
        logger.warning(f"Image not found for task {item['id']}: {item['image_path']}")
        self.stats['missing_images'] += 1
        return None
    
    def _classify_question(self, question: str) -> str:
        """Classify question type based on keywords"""
        q_lower = question.lower()
        
        # Numerical questions
        if any(word in q_lower for word in ['how many', 'how much', 'what is the value', 'total']):
            return 'numerical'
        
        # Comparison questions
        if any(word in q_lower for word in ['compare', 'difference', 'higher', 'lower', 'more', 'less']):
            return 'comparison'
        
        # Trend questions
        if any(word in q_lower for word in ['trend', 'increase', 'decrease', 'growth', 'decline']):
            return 'trend'
        
        # Min/Max questions
        if any(word in q_lower for word in ['highest', 'lowest', 'maximum', 'minimum', 'largest', 'smallest']):
            return 'minmax'
        
        # Data retrieval
        if any(word in q_lower for word in ['what', 'which', 'when', 'where']):
            return 'retrieval'
        
        return 'other'
    
    def get_task_ids(
        self,
        question_type: Optional[str] = None,
        image_filter: Optional[str] = None,
        shuffle: bool = False,
        seed: Optional[int] = None
    ) -> List[str]:
        """
        Get task IDs with optional filtering
        
        Args:
            question_type: Filter by question type (numerical, comparison, etc.)
            image_filter: Filter by image path pattern
            shuffle: Whether to shuffle the results
            seed: Random seed for shuffling
            
        Returns:
            List of task IDs
        """
        # Start with all tasks
        task_ids = list(self._task_index.keys())
        
        # Apply question type filter
        if question_type:
            if question_type in self.question_types:
                task_ids = [tid for tid in task_ids if tid in self.question_types[question_type]]
            else:
                logger.warning(f"Unknown question type: {question_type}")
                logger.info(f"Available types: {list(self.question_types.keys())}")
        
        # Apply image filter
        if image_filter:
            task_ids = [
                tid for tid in task_ids 
                if image_filter in self._task_index[tid]['image_path']
            ]
        
        # Shuffle if requested
        if shuffle:
            if seed is not None:
                random.seed(seed)
            random.shuffle(task_ids)
        
        return task_ids
    
    def get_task_data(self, task_id: str) -> Dict[str, Any]:
        """
        Get complete task data with caching
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task data dictionary
        """
        # Check cache first
        if task_id in self._cache:
            # Move to end (LRU)
            self._cache_order.remove(task_id)
            self._cache_order.append(task_id)
            return self._cache[task_id]
        
        # Get from index
        if task_id not in self._task_index:
            raise ValueError(f"Task ID not found: {task_id}")
        
        task_data = self._task_index[task_id].copy()
        
        # Add to cache
        self._add_to_cache(task_id, task_data)
        
        return task_data
    
    def _add_to_cache(self, task_id: str, task_data: Dict[str, Any]):
        """Add task to cache with LRU eviction"""
        if len(self._cache) >= self.cache_size:
            # Evict oldest
            oldest_id = self._cache_order.pop(0)
            del self._cache[oldest_id]
        
        self._cache[task_id] = task_data
        self._cache_order.append(task_id)
    
    def get_batch(self, task_ids: List[str]) -> List[Dict[str, Any]]:
        """Get multiple tasks efficiently"""
        return [self.get_task_data(tid) for tid in task_ids]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about the dataset"""
        stats = dict(self.stats)
        
        # Add question type distribution
        stats['question_type_distribution'] = {
            qtype: len(task_ids) 
            for qtype, task_ids in self.question_types.items()
        }
        
        # Add image reuse statistics
        image_reuse = [len(tasks) for tasks in self.image_index.values()]
        if image_reuse:
            stats['unique_images'] = len(self.image_index)
            stats['avg_questions_per_image'] = sum(image_reuse) / len(image_reuse)
            stats['max_questions_per_image'] = max(image_reuse)
        
        return stats
    
    def _log_statistics(self):
        """Log dataset statistics"""
        stats = self.get_statistics()
        logger.info("ChartQA Dataset Statistics:")
        logger.info(f"  Total tasks: {stats['total']}")
        logger.info(f"  Unique images: {stats.get('unique_images', 'N/A')}")
        
        if 'question_type_distribution' in stats:
            logger.info("  Question types:")
            for qtype, count in stats['question_type_distribution'].items():
                logger.info(f"    {qtype}: {count}")
    
    def sample_tasks(
        self,
        n: int,
        stratified: bool = False,
        seed: Optional[int] = None
    ) -> List[str]:
        """
        Sample tasks from the dataset
        
        Args:
            n: Number of tasks to sample
            stratified: Whether to sample proportionally from each question type
            seed: Random seed
            
        Returns:
            List of sampled task IDs
        """
        if seed is not None:
            random.seed(seed)
        
        if not stratified:
            # Simple random sampling
            all_ids = list(self._task_index.keys())
            return random.sample(all_ids, min(n, len(all_ids)))
        
        # Stratified sampling
        sampled = []
        total_tasks = len(self._task_index)
        
        for qtype, task_ids in self.question_types.items():
            # Calculate proportion
            proportion = len(task_ids) / total_tasks
            n_sample = int(n * proportion)
            
            # Sample from this question type
            sampled.extend(random.sample(task_ids, min(n_sample, len(task_ids))))
        
        # If we need more samples, add randomly
        if len(sampled) < n:
            remaining = set(self._task_index.keys()) - set(sampled)
            additional = min(n - len(sampled), len(remaining))
            sampled.extend(random.sample(list(remaining), additional))
        
        return sampled[:n]
    
    def get_hard_examples(self, n: int = 10) -> List[str]:
        """Get examples that are typically harder (comparison, calculation)"""
        hard_types = ['comparison', 'trend', 'numerical']
        hard_ids = []
        
        for qtype in hard_types:
            hard_ids.extend(self.question_types.get(qtype, []))
        
        return random.sample(hard_ids, min(n, len(hard_ids)))
    
    def get_examples_by_image(self, image_path: str) -> List[str]:
        """Get all tasks that use a specific image"""
        return self.image_index.get(image_path, [])
    
    def validate_all(self) -> Dict[str, List[str]]:
        """Validate all tasks and return issues"""
        issues = defaultdict(list)
        
        for task_id, task_data in self._task_index.items():
            # Check image exists
            if not Path(task_data['image_path']).exists():
                issues['missing_image'].append(task_id)
            
            # Check answer format
            if not task_data.get('answer'):
                issues['missing_answer'].append(task_id)
            
            # Check question length
            if len(task_data.get('question', '')) < 5:
                issues['short_question'].append(task_id)
        
        return dict(issues)
