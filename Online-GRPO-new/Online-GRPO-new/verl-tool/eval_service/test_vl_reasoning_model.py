#!/usr/bin/env python
"""
model_evaluate.py - 模型评估脚本
用于测试verl-tool训练的VL模型
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
import re
from typing import Dict, List, Optional, Any
import argparse
from tqdm import tqdm
import traceback

from openai import OpenAI


class ModelEvaluator:
    def __init__(self, 
                 model_path: str,
                 api_base_url: str = "http://localhost:5000/v1",
                 output_dir: str = "./evaluation_results",
                 max_retries: int = 3,
                 retry_delay: int = 5):
        """
        初始化评估器
        
        Args:
            model_path: 模型路径（用作model参数）
            api_base_url: API服务地址
            output_dir: 结果保存目录
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
        """
        self.model_path = model_path
        self.api_base_url = api_base_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            base_url=api_base_url,
            api_key="not-needed"  # API key不需要，但参数必须有
        )
        
        # 进度文件
        self.progress_file = self.output_dir / "progress.json"
        self.results_file = self.output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # 加载已完成的任务
        self.completed_tasks = self.load_progress()
        
    def load_progress(self) -> set:
        """加载已完成的任务ID"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
                return set(data.get('completed_tasks', []))
        return set()
    
    def save_progress(self, task_id: str):
        """保存进度"""
        self.completed_tasks.add(task_id)
        with open(self.progress_file, 'w') as f:
            json.dump({'completed_tasks': list(self.completed_tasks)}, f)
    
    def extract_answer(self, response: str) -> str:
        """
        从响应中提取答案
        
        优先级：
        1. <answer>标签内容
        2. 最后一行非空内容
        3. 完整响应
        """
        # 尝试提取<answer>标签
        answer_pattern = r'<answer>(.*?)</answer>'
        matches = re.findall(answer_pattern, response, re.DOTALL)
        if matches:
            return matches[-1].strip()  # 返回最后一个匹配
        
        # 尝试提取最后的非空行
        lines = response.strip().split('\n')
        for line in reversed(lines):
            if line.strip():
                return line.strip()
        
        # 返回完整响应
        return response.strip()
    
    def format_messages(self, item: Dict) -> List[Dict]:
        """
        格式化消息用于API调用
        根据test_chartqa_api.py的格式
        """
        messages_formatted = []
        
        for msg in item['messages']:
            role = msg['role']
            content = msg['content']
            
            if role == 'system':
                # System消息总是纯文本
                if isinstance(content, str):
                    messages_formatted.append({
                        'role': 'system',
                        'content': content
                    })
                elif isinstance(content, list):
                    # 提取文本内容
                    text_parts = [item['text'] for item in content if item.get('type') == 'text']
                    messages_formatted.append({
                        'role': 'system',
                        'content': ' '.join(text_parts)
                    })
            
            elif role == 'user':
                # User消息可能包含图片
                if isinstance(content, str):
                    messages_formatted.append({
                        'role': 'user',
                        'content': content
                    })
                elif isinstance(content, list):
                    # 构建多模态内容
                    formatted_content = []
                    for item_content in content:
                        if item_content.get('type') == 'text':
                            formatted_content.append({
                                'type': 'text',
                                'text': item_content.get('text', '')
                            })
                        elif item_content.get('type') == 'image':
                            # 使用图片路径
                            formatted_content.append({
                                'type': 'image',
                                'image': item['image_path']  # 使用item中的image_path
                            })
                    
                    messages_formatted.append({
                        'role': 'user',
                        'content': formatted_content
                    })
        
        return messages_formatted
    
    def call_model(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 2048) -> Optional[str]:
        """
        调用模型API
        
        Returns:
            模型响应或None（如果失败）
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_path,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            
            except Exception as e:
                print(f"  Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    print(f"  All retries failed")
                    return None
    
    def evaluate_single(self, item: Dict) -> Dict:
        """
        评估单个样本
        
        Returns:
            包含结果的字典
        """
        task_id = item['task_id']
        
        # 检查是否已完成
        if task_id in self.completed_tasks:
            print(f"  Skipping {task_id} (already completed)")
            return None
        
        print(f"  Processing {task_id}...")
        
        # 格式化消息
        messages = self.format_messages(item)
        
        # 调用模型
        start_time = time.time()
        response = self.call_model(messages)
        elapsed_time = time.time() - start_time
        
        # 构建结果
        result = {
            'task_id': task_id,
            'task_type': item.get('task_type', 'unknown'),
            'dataset': item.get('dataset', 'unknown'),
            'ground_truth': item.get('metadata', {}).get('ground_truth', ''),
            'has_choices': item.get('metadata', {}).get('has_choices', False),
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': elapsed_time,
            'retry_count': 0  # TODO: 从call_model返回实际重试次数
        }
        
        if response is not None:
            result['status'] = 'success'
            result['full_response'] = response
            result['extracted_answer'] = self.extract_answer(response)
        else:
            result['status'] = 'failed'
            result['full_response'] = ''
            result['extracted_answer'] = ''
            result['error'] = 'Max retries exceeded'
        
        return result
    
    def save_results(self, results: List[Dict], statistics: Dict):
        """保存结果到文件"""
        output = {
            'metadata': {
                'model_path': self.model_path,
                'api_base_url': self.api_base_url,
                'timestamp': datetime.now().isoformat(),
                'total_samples': statistics['total'],
                'completed_samples': statistics['completed'],
                'failed_samples': statistics['failed']
            },
            'statistics_by_task': statistics['by_task'],
            'results': results
        }
        
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to: {self.results_file}")
    
    def evaluate(self, test_file: str, num_samples: int = None):
        """
        执行评估
        
        Args:
            test_file: 测试数据文件路径
            num_samples: 要测试的样本数（None表示全部）
        """
        print(f"Loading test data from: {test_file}")
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        # 限制样本数
        if num_samples is not None:
            test_data = test_data[:num_samples]
        
        print(f"Total samples to evaluate: {len(test_data)}")
        
        # 统计信息
        statistics = {
            'total': len(test_data),
            'completed': 0,
            'failed': 0,
            'by_task': {}
        }
        
        results = []
        
        # 使用tqdm显示进度
        with tqdm(total=len(test_data), desc="Evaluating") as pbar:
            for item in test_data:
                task_type = item.get('task_type', 'unknown')
                
                # 评估单个样本
                try:
                    result = self.evaluate_single(item)
                    
                    if result is not None:
                        results.append(result)
                        
                        # 更新统计
                        if result['status'] == 'success':
                            statistics['completed'] += 1
                            # 保存进度
                            self.save_progress(result['task_id'])
                        else:
                            statistics['failed'] += 1
                        
                        # 按任务类型统计
                        if task_type not in statistics['by_task']:
                            statistics['by_task'][task_type] = {'total': 0, 'completed': 0, 'failed': 0}
                        
                        statistics['by_task'][task_type]['total'] += 1
                        if result['status'] == 'success':
                            statistics['by_task'][task_type]['completed'] += 1
                        else:
                            statistics['by_task'][task_type]['failed'] += 1
                        
                        # 实时保存结果（防止中断丢失）
                        if len(results) % 10 == 0:
                            self.save_results(results, statistics)
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'completed': statistics['completed'],
                        'failed': statistics['failed']
                    })
                
                except Exception as e:
                    print(f"\nError processing {item.get('task_id', 'unknown')}: {str(e)}")
                    traceback.print_exc()
                    statistics['failed'] += 1
                    pbar.update(1)
        
        # 最终保存
        self.save_results(results, statistics)
        
        # 打印统计信息
        print("\n" + "="*50)
        print("Evaluation Complete!")
        print("="*50)
        print(f"Total: {statistics['total']}")
        print(f"Completed: {statistics['completed']}")
        print(f"Failed: {statistics['failed']}")
        print(f"Success Rate: {statistics['completed']/statistics['total']*100:.2f}%")
        
        print("\nBy Task Type:")
        for task_type, stats in statistics['by_task'].items():
            print(f"  {task_type}:")
            print(f"    Total: {stats['total']}")
            print(f"    Completed: {stats['completed']}")
            print(f"    Failed: {stats['failed']}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate VL model')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the model (e.g., /path/to/global_step_200/actor/huggingface)')
    parser.add_argument('--test-file', type=str, required=True,
                        help='Path to test dataset JSON file')
    parser.add_argument('--api-base-url', type=str, default='http://localhost:5000/v1',
                        help='API base URL')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples to test (default: 100, use -1 for all)')
    parser.add_argument('--max-retries', type=int, default=3,
                        help='Maximum number of retries per sample')
    
    args = parser.parse_args()
    
    # 处理num_samples参数
    num_samples = None if args.num_samples == -1 else args.num_samples
    
    # 创建评估器
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        api_base_url=args.api_base_url,
        output_dir=args.output_dir,
        max_retries=args.max_retries
    )
    
    # 执行评估
    evaluator.evaluate(
        test_file=args.test_file,
        num_samples=num_samples
    )


if __name__ == '__main__':
    main()