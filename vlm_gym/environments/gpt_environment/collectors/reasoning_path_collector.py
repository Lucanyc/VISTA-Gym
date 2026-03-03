# vlm_gym/environments/gpt_environment/collectors/reasoning_path_collector.py

"""
Reasoning Path Collector for GPT Environment
Collects, filters, and stores high-quality reasoning paths for VLM training
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
import logging
import hashlib
import pandas as pd
from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)


@dataclass
class ReasoningPath:
    """A single reasoning path with metadata"""
    # Identifiers
    path_id: str
    dialogue_id: str
    task_id: str
    task_type: str
    
    # Core content
    dialogue: List[Dict[str, Any]]
    reasoning_chain: List[Dict[str, Any]]
    task_info: Dict[str, Any]
    
    # Quality metrics
    quality_score: float
    quality_breakdown: Dict[str, float]
    
    # Outcome
    success: bool
    final_answer: Any
    correct_answer: Any
    turns_taken: int
    time_taken: float
    
    # Metadata
    timestamp: str
    student_profile: Dict[str, Any] = field(default_factory=dict)
    teaching_strategy: str = ""
    key_insights: List[str] = field(default_factory=list)
    
    # Filtering flags
    is_high_quality: bool = False
    has_clear_reasoning: bool = False
    has_complete_solution: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    def to_training_format(self) -> Dict[str, Any]:
        """Format for model training"""
        # Extract clean conversation for training
        conversation = []
        for turn in self.dialogue:
            if turn['role'] in ['gpt_teacher', 'environment']:
                role = 'assistant'
            elif turn['role'] == 'agent':
                role = 'user'
            else:
                continue
                
            conversation.append({
                'role': role,
                'content': turn['content']
            })
        
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'conversation': conversation,
            'reasoning_steps': [step['content'] for step in self.reasoning_chain],
            'success': self.success,
            'quality_score': self.quality_score,
            'metadata': {
                'turns': self.turns_taken,
                'strategy': self.teaching_strategy,
                'has_complete_solution': self.has_complete_solution
            }
        }


class ReasoningPathCollector:
    """
    Collects and manages high-quality reasoning paths
    
    Key responsibilities:
    1. Filter dialogues based on quality criteria
    2. Store reasoning paths in multiple formats
    3. Maintain statistics and quality metrics
    4. Export data for training
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize collector
        
        Args:
            config: Configuration with:
                - save_path: Base directory for saving paths
                - quality_threshold: Minimum quality score
                - formats: List of export formats ['json', 'hf_dataset', 'csv']
                - buffer_size: Number of paths to buffer before saving
                - task_balanced: Whether to balance across task types
        """
        self.save_path = Path(config.get('save_path', './reasoning_paths'))
        self.quality_threshold = config.get('quality_threshold', 0.7)
        self.formats = config.get('formats', ['json', 'hf_dataset'])
        self.buffer_size = config.get('buffer_size', 100)
        self.task_balanced = config.get('task_balanced', True)
        
        # Create directory structure
        self._setup_directories()
        
        # Collection buffers
        self.path_buffer: List[ReasoningPath] = []
        self.collected_paths: Dict[str, List[ReasoningPath]] = defaultdict(list)
        
        # Statistics
        self.stats = {
            'total_evaluated': 0,
            'total_collected': 0,
            'collected_by_task': defaultdict(int),
            'quality_distribution': defaultdict(int),
            'success_rate': 0.0,
            'avg_quality_score': 0.0,
            'avg_turns': 0.0
        }
        
        # Load existing statistics if available
        self._load_statistics()
    
    def _setup_directories(self):
        """Create directory structure for saving paths"""
        # Base directory
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Task-specific directories
        self.task_dirs = {}
        for task_type in ['figureqa', 'chartqa', 'clevr', 'geometry3k', 'geoqa', 
                         'iconqa', 'scienceqa', 'mathvista', 'olympiadbench']:
            task_dir = self.save_path / task_type
            task_dir.mkdir(exist_ok=True)
            self.task_dirs[task_type] = task_dir
            
            # Quality subdirectories
            (task_dir / 'high_quality').mkdir(exist_ok=True)
            (task_dir / 'all').mkdir(exist_ok=True)
        
        # Statistics file
        self.stats_file = self.save_path / 'collection_statistics.json'
        
        # Metadata directory
        (self.save_path / 'metadata').mkdir(exist_ok=True)
    
    def collect_path(self, dialogue: List[Dict[str, Any]], task: Any,
                    outcome: Dict[str, Any], quality_metrics: Dict[str, float]):
        """
        Collect a reasoning path if it meets quality criteria
        
        Args:
            dialogue: Complete dialogue history
            task: Task object
            outcome: Episode outcome information
            quality_metrics: Quality assessment scores
        """
        self.stats['total_evaluated'] += 1
        
        # Extract task info
        task_info = self._extract_task_info(task)
        
        # Check quality threshold
        overall_quality = quality_metrics.get('overall', 0.0)
        if overall_quality < self.quality_threshold:
            logger.debug(f"Path rejected: quality {overall_quality:.2f} < {self.quality_threshold}")
            return
        
        # Extract reasoning chain
        reasoning_chain = self._extract_reasoning_chain(dialogue)
        
        # Create reasoning path object
        path = ReasoningPath(
            path_id=self._generate_path_id(task_info['id'], outcome),
            dialogue_id=f"{task_info['id']}_{int(time.time())}",
            task_id=task_info['id'],
            task_type=task_info['type'],
            dialogue=dialogue,
            reasoning_chain=reasoning_chain,
            task_info=task_info,
            quality_score=overall_quality,
            quality_breakdown=quality_metrics,
            success=outcome.get('success', False),
            final_answer=self._extract_final_answer(dialogue),
            correct_answer=task_info.get('answer'),
            turns_taken=outcome.get('total_turns', len(dialogue)),
            time_taken=outcome.get('duration', 0.0),
            timestamp=datetime.now().isoformat(),
            teaching_strategy=outcome.get('strategy_used', 'unknown'),
            key_insights=self._extract_key_insights(dialogue)
        )
        
        # Set quality flags
        path.is_high_quality = overall_quality >= 0.8
        path.has_clear_reasoning = quality_metrics.get('reasoning_clarity', 0) >= 0.7
        path.has_complete_solution = (
            path.success and 
            quality_metrics.get('step_completeness', 0) >= 0.8
        )
        
        # Add to buffer
        self.path_buffer.append(path)
        self.collected_paths[task_info['type']].append(path)
        
        # Update statistics
        self._update_statistics(path)
        
        # Save if buffer is full
        if len(self.path_buffer) >= self.buffer_size:
            self.save_buffered_paths()
        
        logger.info(f"Collected reasoning path: {path.path_id} (quality: {overall_quality:.2f})")
    
    def _extract_task_info(self, task: Any) -> Dict[str, Any]:
        """Extract task information"""
        # Try unified interface first
        if hasattr(task, 'get_gpt_teacher_info'):
            return task.get_gpt_teacher_info()
        
        # Fallback to manual extraction
        return {
            'id': getattr(task, 'task_id', 'unknown'),
            'type': self._infer_task_type(task),
            'answer': getattr(task, 'answer', None),
            'question': getattr(task, 'question', ''),
            'metadata': getattr(task, 'metadata', {})
        }
    
    def _infer_task_type(self, task: Any) -> str:
        """Infer task type from task object"""
        class_name = task.__class__.__name__.lower()
        for task_type in self.task_dirs.keys():
            if task_type in class_name:
                return task_type
        return 'unknown'
    
    def _extract_reasoning_chain(self, dialogue: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract reasoning steps from dialogue"""
        reasoning_chain = []
        
        for turn in dialogue:
            if turn.get('role') == 'agent':
                # Check if turn contains reasoning
                metadata = turn.get('metadata', {})
                if 'reasoning_type' in metadata:
                    reasoning_chain.append({
                        'turn_id': turn.get('turn_id', len(reasoning_chain)),
                        'content': turn['content'],
                        'reasoning_type': metadata['reasoning_type'],
                        'confidence': metadata.get('confidence', 0.5)
                    })
                elif self._contains_reasoning(turn['content']):
                    # Fallback: detect reasoning from content
                    reasoning_chain.append({
                        'turn_id': turn.get('turn_id', len(reasoning_chain)),
                        'content': turn['content'],
                        'reasoning_type': 'general',
                        'confidence': 0.5
                    })
        
        return reasoning_chain
    
    def _contains_reasoning(self, content: str) -> bool:
        """Check if content contains reasoning indicators"""
        reasoning_keywords = [
            'because', 'therefore', 'since', 'thus', 'i see', 'i notice',
            'this means', 'which suggests', 'based on', 'observe'
        ]
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in reasoning_keywords)
    
    def _extract_final_answer(self, dialogue: List[Dict[str, Any]]) -> Optional[str]:
        """Extract final answer from dialogue"""
        # Look for the last agent turn
        for turn in reversed(dialogue):
            if turn.get('role') == 'agent':
                content = turn['content']
                # Look for answer patterns
                import re
                patterns = [
                    r"(?:the answer is|my answer is) (.+?)(?:\.|$)",
                    r"(?:therefore|so|thus) (.+?) is the answer",
                    r"\b(yes|no)\b",
                    r"= (.+?)(?:\.|$)"
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, content.lower())
                    if match:
                        return match.group(1) if match.groups() else match.group(0)
                
                # Return last sentence as fallback
                sentences = content.split('.')
                if sentences:
                    return sentences[-1].strip()
        
        return None
    
    def _extract_key_insights(self, dialogue: List[Dict[str, Any]]) -> List[str]:
        """Extract key insights from dialogue"""
        insights = []
        
        for turn in dialogue:
            if turn.get('role') == 'agent':
                metadata = turn.get('metadata', {})
                if metadata.get('is_key_insight'):
                    insights.append(turn['content'])
        
        return insights
    
    def _generate_path_id(self, task_id: str, outcome: Dict[str, Any]) -> str:
        """Generate unique path ID"""
        # Create unique identifier
        unique_string = f"{task_id}_{outcome.get('total_turns', 0)}_{time.time()}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:12]
    
    def _update_statistics(self, path: ReasoningPath):
        """Update collection statistics"""
        self.stats['total_collected'] += 1
        self.stats['collected_by_task'][path.task_type] += 1
        
        # Quality distribution
        quality_bucket = int(path.quality_score * 10) / 10  # Round to 0.1
        self.stats['quality_distribution'][quality_bucket] += 1
        
        # Running averages
        n = self.stats['total_collected']
        self.stats['avg_quality_score'] = (
            (self.stats['avg_quality_score'] * (n - 1) + path.quality_score) / n
        )
        self.stats['avg_turns'] = (
            (self.stats['avg_turns'] * (n - 1) + path.turns_taken) / n
        )
        
        # Success rate
        if path.success:
            success_count = self.stats.get('success_count', 0) + 1
            self.stats['success_count'] = success_count
            self.stats['success_rate'] = success_count / n
    
    def save_buffered_paths(self):
        """Save buffered paths to disk"""
        if not self.path_buffer:
            return
        
        logger.info(f"Saving {len(self.path_buffer)} buffered paths...")
        
        for format_type in self.formats:
            if format_type == 'json':
                self._save_as_json()
            elif format_type == 'hf_dataset':
                self._save_as_hf_dataset()
            elif format_type == 'csv':
                self._save_as_csv()
        
        # Clear buffer after saving
        self.path_buffer.clear()
        
        # Save updated statistics
        self._save_statistics()
    
    def _save_as_json(self):
        """Save paths as JSON files"""
        for path in self.path_buffer:
            # Determine save location
            quality_dir = 'high_quality' if path.is_high_quality else 'all'
            save_dir = self.task_dirs[path.task_type] / quality_dir
            
            # Save individual path
            file_path = save_dir / f"{path.path_id}.json"
            with open(file_path, 'w') as f:
                json.dump(path.to_dict(), f, indent=2)
    
    def _save_as_hf_dataset(self):
        """Save paths as HuggingFace dataset"""
        # Group by task type
        task_datasets = defaultdict(list)
        
        for path in self.path_buffer:
            training_format = path.to_training_format()
            task_datasets[path.task_type].append(training_format)
        
        # Save each task type as a dataset
        for task_type, data in task_datasets.items():
            if data:
                dataset = Dataset.from_list(data)
                dataset_path = self.task_dirs[task_type] / 'hf_dataset'
                dataset.save_to_disk(str(dataset_path))
    
    def _save_as_csv(self):
        """Save paths summary as CSV"""
        # Create summary records
        records = []
        for path in self.path_buffer:
            records.append({
                'path_id': path.path_id,
                'task_id': path.task_id,
                'task_type': path.task_type,
                'success': path.success,
                'quality_score': path.quality_score,
                'turns': path.turns_taken,
                'time_seconds': path.time_taken,
                'strategy': path.teaching_strategy,
                'timestamp': path.timestamp
            })
        
        # Append to existing CSV
        csv_path = self.save_path / 'reasoning_paths_summary.csv'
        df = pd.DataFrame(records)
        
        if csv_path.exists():
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current collection statistics"""
        return {
            'total_evaluated': self.stats['total_evaluated'],
            'total_collected': self.stats['total_collected'],
            'collection_rate': (
                self.stats['total_collected'] / max(1, self.stats['total_evaluated'])
            ),
            'by_task_type': dict(self.stats['collected_by_task']),
            'quality_distribution': dict(self.stats['quality_distribution']),
            'average_quality': self.stats['avg_quality_score'],
            'average_turns': self.stats['avg_turns'],
            'success_rate': self.stats['success_rate'],
            'buffer_size': len(self.path_buffer)
        }
    
    def get_collected_count(self) -> int:
        """Get total number of collected paths"""
        return self.stats['total_collected']
    
    def _save_statistics(self):
        """Save statistics to file"""
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def _load_statistics(self):
        """Load existing statistics if available"""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r') as f:
                    saved_stats = json.load(f)
                    self.stats.update(saved_stats)
                logger.info(f"Loaded existing statistics: {self.stats['total_collected']} paths collected")
            except Exception as e:
                logger.error(f"Failed to load statistics: {e}")
    
    def export_dataset(self, output_path: str, format_type: str = 'hf_dataset',
                      min_quality: float = None, task_types: List[str] = None) -> Dict[str, Any]:
        """
        Export collected paths as a dataset
        
        Args:
            output_path: Where to save the dataset
            format_type: Export format ('hf_dataset', 'json', 'csv')
            min_quality: Minimum quality score filter
            task_types: List of task types to include
            
        Returns:
            Export summary
        """
        min_quality = min_quality or self.quality_threshold
        task_types = task_types or list(self.task_dirs.keys())
        
        # Collect all paths matching criteria
        all_paths = []
        for task_type in task_types:
            if task_type in self.collected_paths:
                for path in self.collected_paths[task_type]:
                    if path.quality_score >= min_quality:
                        all_paths.append(path)
        
        # Also check saved paths
        for task_type in task_types:
            task_dir = self.task_dirs.get(task_type)
            if task_dir:
                json_files = list((task_dir / 'high_quality').glob('*.json'))
                if min_quality < 0.8:
                    json_files.extend(list((task_dir / 'all').glob('*.json')))
                
                for json_file in json_files:
                    try:
                        with open(json_file, 'r') as f:
                            path_data = json.load(f)
                            if path_data['quality_score'] >= min_quality:
                                # Reconstruct ReasoningPath (simplified)
                                all_paths.append(path_data)
                    except Exception as e:
                        logger.error(f"Failed to load {json_file}: {e}")
        
        # Export based on format
        if format_type == 'hf_dataset':
            # Create HuggingFace dataset
            dataset_records = []
            for path in all_paths:
                if isinstance(path, ReasoningPath):
                    dataset_records.append(path.to_training_format())
                else:
                    # Already in dict format
                    dataset_records.append(path)
            
            dataset = Dataset.from_list(dataset_records)
            
            # Split by task type if requested
            if len(task_types) > 1:
                dataset_dict = DatasetDict({
                    task_type: dataset.filter(lambda x: x['task_type'] == task_type)
                    for task_type in task_types
                })
                dataset_dict.save_to_disk(output_path)
            else:
                dataset.save_to_disk(output_path)
            
            return {
                'format': 'hf_dataset',
                'total_examples': len(dataset_records),
                'task_types': task_types,
                'min_quality': min_quality,
                'output_path': output_path
            }
        
        elif format_type == 'json':
            # Save as JSON
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump([
                    path.to_dict() if isinstance(path, ReasoningPath) else path
                    for path in all_paths
                ], f, indent=2)
            
            return {
                'format': 'json',
                'total_examples': len(all_paths),
                'output_path': output_path
            }
    
    def get_sample_paths(self, n: int = 5, task_type: Optional[str] = None,
                        min_quality: float = 0.8) -> List[ReasoningPath]:
        """Get sample high-quality paths for inspection"""
        sample_paths = []
        
        if task_type:
            candidates = [p for p in self.collected_paths.get(task_type, [])
                         if p.quality_score >= min_quality]
        else:
            candidates = []
            for paths in self.collected_paths.values():
                candidates.extend([p for p in paths if p.quality_score >= min_quality])
        
        # Sort by quality and return top n
        candidates.sort(key=lambda p: p.quality_score, reverse=True)
        return candidates[:n]
    
    def close(self):
        """Save any remaining buffered paths and close collector"""
        if self.path_buffer:
            self.save_buffered_paths()
        
        # Final statistics save
        self._save_statistics()
        
        # Log summary
        logger.info(f"Reasoning Path Collector closed. Total collected: {self.stats['total_collected']}")