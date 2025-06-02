import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd

class MetricsTracker:
    """Track and compute various metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.episode_metrics = []
        
    def add(self, name: str, value: float):
        """Add a single metric value"""
        self.metrics[name].append(value)
        
    def add_batch(self, metrics_dict: Dict[str, float]):
        """Add multiple metrics at once"""
        for name, value in metrics_dict.items():
            self.add(name, value)
            
    def get_mean(self, name: str) -> float:
        """Get mean value of a metric"""
        values = self.metrics.get(name, [])
        return np.mean(values) if values else 0.0
        
    def get_std(self, name: str) -> float:
        """Get standard deviation of a metric"""
        values = self.metrics.get(name, [])
        return np.std(values) if values else 0.0
        
    def get_all_means(self) -> Dict[str, float]:
        """Get mean values for all metrics"""
        return {name: self.get_mean(name) for name in self.metrics}
        
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics"""
        summary = {}
        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "count": len(values)
                }
        return summary
        
    def save_episode(self):
        """Save current metrics as an episode"""
        episode_summary = self.get_all_means()
        self.episode_metrics.append(episode_summary)
        self.reset()
        
    def reset(self):
        """Reset current metrics"""
        self.metrics.clear()
        
    def to_dataframe(self) -> pd.DataFrame:
        """Convert episode metrics to pandas DataFrame"""
        return pd.DataFrame(self.episode_metrics)

def compute_vision_qa_metrics(
    predictions: List[str],
    ground_truths: List[str],
    question_types: Optional[List[str]] = None
) -> Dict[str, float]:
    """Compute metrics for vision QA tasks"""
    metrics = {}
    
    # Overall accuracy
    correct = sum(pred.lower().strip() == gt.lower().strip() 
                 for pred, gt in zip(predictions, ground_truths))
    metrics['accuracy'] = correct / len(predictions)
    
    # Per question type accuracy
    if question_types:
        type_correct = defaultdict(int)
        type_total = defaultdict(int)
        
        for pred, gt, qtype in zip(predictions, ground_truths, question_types):
            type_total[qtype] += 1
            if pred.lower().strip() == gt.lower().strip():
                type_correct[qtype] += 1
                
        for qtype in type_total:
            metrics[f'accuracy_{qtype}'] = type_correct[qtype] / type_total[qtype]
            
    return metrics

def compute_classification_metrics(
    predictions: List[int],
    ground_truths: List[int],
    labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Compute classification metrics"""
    
    # Basic metrics
    metrics = {
        'accuracy': accuracy_score(ground_truths, predictions)
    }
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        ground_truths, predictions, average=None
    )
    
    # Macro and weighted averages
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        ground_truths, predictions, average='macro'
    )
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        ground_truths, predictions, average='weighted'
    )
    
    metrics.update({
        'macro_precision': macro_prec,
        'macro_recall': macro_rec,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_prec,
        'weighted_recall': weighted_rec,
        'weighted_f1': weighted_f1,
    })
    
    # Per-class results
    if labels:
        per_class = {}
        for i, label in enumerate(labels):
            per_class[label] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': support[i]
            }
        metrics['per_class'] = per_class
        
    return metrics

def compute_retrieval_metrics(
    retrieved_items: List[List[int]],
    relevant_items: List[List[int]],
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """Compute retrieval metrics (Precision@K, Recall@K, etc.)"""
    metrics = {}
    
    for k in k_values:
        precisions = []
        recalls = []
        
        for retrieved, relevant in zip(retrieved_items, relevant_items):
            retrieved_k = set(retrieved[:k])
            relevant_set = set(relevant)
            
            if len(retrieved_k) > 0:
                precision = len(retrieved_k & relevant_set) / len(retrieved_k)
                precisions.append(precision)
                
            if len(relevant_set) > 0:
                recall = len(retrieved_k & relevant_set) / len(relevant_set)
                recalls.append(recall)
                
        metrics[f'precision@{k}'] = np.mean(precisions)
        metrics[f'recall@{k}'] = np.mean(recalls)
        
    return metrics

def compute_generation_metrics(
    generated_texts: List[str],
    reference_texts: List[str]
) -> Dict[str, float]:
    """Compute text generation metrics"""
    from nltk.translate.bleu_score import sentence_bleu
    from rouge import Rouge
    
    metrics = {}
    
    # BLEU scores
    bleu_scores = []
    for gen, ref in zip(generated_texts, reference_texts):
        score = sentence_bleu([ref.split()], gen.split())
        bleu_scores.append(score)
    metrics['bleu'] = np.mean(bleu_scores)
    
    # ROUGE scores
    rouge = Rouge()
    rouge_scores = rouge.get_scores(generated_texts, reference_texts, avg=True)
    
    metrics.update({
        'rouge-1-f': rouge_scores['rouge-1']['f'],
        'rouge-2-f': rouge_scores['rouge-2']['f'],
        'rouge-l-f': rouge_scores['rouge-l']['f'],
    })
    
    return metrics

def compute_efficiency_metrics(
    episode_data: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Compute efficiency-related metrics"""
    metrics = {}
    
    # Time efficiency
    total_time = sum(step.get('execution_time', 0) for step in episode_data)
    metrics['total_time'] = total_time
    metrics['avg_step_time'] = total_time / len(episode_data) if episode_data else 0
    
    # Action efficiency
    action_types = [step.get('action_type', 'unknown') for step in episode_data]
    unique_actions = len(set(action_types))
    metrics['action_diversity'] = unique_actions / len(action_types) if action_types else 0
    
    # Success efficiency
    successful_steps = sum(1 for step in episode_data if step.get('success', False))
    metrics['success_rate'] = successful_steps / len(episode_data) if episode_data else 0
    
    return metrics
