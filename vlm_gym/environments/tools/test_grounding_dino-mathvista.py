#!/usr/bin/env python3
"""
在 MathVista 数据集上测试 Grounding DINO 工具
"""
import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Any
from PIL import Image
import pandas as pd
from tqdm import tqdm

# 添加项目路径
sys.path.append('/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista')

from vlm_gym.environments.tools.grounding_dino import GroundingDINOTool


class GroundingDINOTester:
    """Grounding DINO 在 MathVista 上的测试器"""
    
    def __init__(self, data_path: str, output_dir: str = None):
        self.data_path = data_path
        self.output_dir = output_dir or f"./grounding_dino_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化工具
        self.tool = GroundingDINOTool({
            'device': 'cuda',
            'box_threshold': 0.35,
            'text_threshold': 0.25
        })
        
        # 统计信息
        self.stats = {
            'total_samples': 0,
            'processed_samples': 0,
            'failed_samples': 0,
            'total_detections': 0,
            'processing_times': [],
            'task_stats': {},
            'error_types': {}
        }
        
    def load_dataset(self) -> List[Dict]:
        """加载 MathVista 数据集"""
        print(f"加载数据集: {self.data_path}")
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        print(f"加载了 {len(data)} 个样本")
        return data
    
    def get_detection_strategy(self, sample: Dict) -> List[Dict]:
        """根据任务类型生成检测策略"""
        task = sample.get('task', '')
        question = sample.get('question', '')
        metadata = sample.get('metadata', {})
        
        strategies = []
        
        # 基础检测 - 所有任务都执行
        strategies.append({
            'name': 'general_objects',
            'caption': 'all objects',
            'box_threshold': 0.3
        })
        
        # 根据任务类型添加特定检测
        if task == 'textbook question answering':
            strategies.extend([
                {'name': 'text_and_formulas', 'caption': 'text and mathematical formulas', 'text_threshold': 0.2},
                {'name': 'graphs_charts', 'caption': 'graphs, charts, and diagrams', 'box_threshold': 0.25},
                {'name': 'numbers', 'caption': 'numbers and numerical values', 'text_threshold': 0.15}
            ])
            
        elif task == 'visual question answering':
            # 分析问题找关键词
            if 'measuring' in question.lower() or 'volume' in question.lower():
                strategies.append({'name': 'measurement', 'caption': 'measuring cup, scale, ruler', 'box_threshold': 0.25})
            if 'count' in question.lower() or 'how many' in question.lower():
                strategies.append({'name': 'counting', 'caption': 'countable objects', 'box_threshold': 0.2})
            
            # 添加常见物体检测
            strategies.extend([
                {'name': 'common_objects', 'caption': 'person, animal, vehicle, furniture', 'box_threshold': 0.3},
                {'name': 'text_in_image', 'caption': 'text, labels, signs', 'text_threshold': 0.2}
            ])
            
        elif task == 'geometry problem solving':
            strategies.extend([
                {'name': 'geometry_shapes', 'caption': 'triangles, circles, rectangles, polygons', 'box_threshold': 0.25},
                {'name': 'geometry_annotations', 'caption': 'angles, points, lines, labels', 'text_threshold': 0.15},
                {'name': 'mathematical_symbols', 'caption': 'mathematical symbols and notations', 'text_threshold': 0.2}
            ])
            
        elif 'chart' in task.lower() or 'plot' in task.lower():
            strategies.extend([
                {'name': 'chart_elements', 'caption': 'bars, lines, points, axes', 'box_threshold': 0.25},
                {'name': 'chart_text', 'caption': 'title, labels, legend, axis labels', 'text_threshold': 0.2}
            ])
            
        return strategies
    
    def process_sample(self, sample: Dict) -> Dict[str, Any]:
        """处理单个样本"""
        sample_id = sample['id']
        image_path = sample['image_path']
        
        result = {
            'id': sample_id,
            'task': sample.get('task', ''),
            'detections': {},
            'errors': [],
            'processing_time': 0
        }
        
        # 检查图像是否存在
        if not os.path.exists(image_path):
            error_msg = f"图像文件不存在: {image_path}"
            result['errors'].append(error_msg)
            print(f"\n错误 - {sample_id}: {error_msg}")
            return result
        
        try:
            # 加载图像
            start_time = time.time()
            self.tool.reset(image_path)
            
            # 获取检测策略
            strategies = self.get_detection_strategy(sample)
            
            # 执行每个检测策略
            for strategy in strategies:
                strategy_name = strategy.get('name')
                strategy_params = {k: v for k, v in strategy.items() if k != 'name'}
                try:
                    detection_result = self.tool.execute(json.dumps(strategy_params))
                    
                    if 'error' not in detection_result:
                        result['detections'][strategy_name] = {
                            'num_detections': detection_result['num_detections'],
                            'phrases': detection_result['phrases'],
                            'logits': detection_result['logits'],
                            'caption': strategy_params.get('caption', '')
                        }
                        self.stats['total_detections'] += detection_result['num_detections']
                    else:
                        error_msg = f"{strategy_name}: {detection_result['error']}"
                        result['errors'].append(error_msg)
                        if 'traceback' in detection_result:
                            print(f"\n错误追踪 - {sample_id}/{strategy_name}:\n{detection_result['traceback']}")
                        
                except Exception as e:
                    error_msg = f"{strategy_name}: {str(e)}"
                    result['errors'].append(error_msg)
                    print(f"\n异常 - {sample_id}/{strategy_name}: {e}")
                    import traceback
                    traceback.print_exc()
            
            result['processing_time'] = time.time() - start_time
            self.stats['processing_times'].append(result['processing_time'])
            
        except Exception as e:
            error_msg = f"处理失败: {str(e)}"
            result['errors'].append(error_msg)
            print(f"\n处理异常 - {sample_id}: {e}")
            import traceback
            traceback.print_exc()
            
        return result
    
    def analyze_results(self, results: List[Dict]) -> Dict:
        """分析测试结果"""
        analysis = {
            'task_performance': {},
            'detection_summary': {},
            'common_objects': {},
            'processing_stats': {}
        }
        
        # 按任务类型分组
        task_groups = {}
        for result in results:
            task = result['task']
            if task not in task_groups:
                task_groups[task] = []
            task_groups[task].append(result)
        
        # 分析每个任务类型
        for task, task_results in task_groups.items():
            task_stats = {
                'total_samples': len(task_results),
                'successful_samples': sum(1 for r in task_results if not r['errors']),
                'avg_detections': 0,
                'avg_processing_time': 0,
                'common_detections': {}
            }
            
            total_detections = 0
            total_time = 0
            all_phrases = []
            
            for result in task_results:
                if not result['errors']:
                    total_time += result['processing_time']
                    for detection in result['detections'].values():
                        total_detections += detection['num_detections']
                        all_phrases.extend(detection['phrases'])
            
            if task_stats['successful_samples'] > 0:
                task_stats['avg_detections'] = total_detections / task_stats['successful_samples']
                task_stats['avg_processing_time'] = total_time / task_stats['successful_samples']
            
            # 统计常见检测对象
            from collections import Counter
            phrase_counter = Counter(all_phrases)
            task_stats['common_detections'] = dict(phrase_counter.most_common(10))
            
            analysis['task_performance'][task] = task_stats
        
        # 整体处理统计
        if self.stats['processing_times']:
            analysis['processing_stats'] = {
                'avg_time': sum(self.stats['processing_times']) / len(self.stats['processing_times']),
                'min_time': min(self.stats['processing_times']),
                'max_time': max(self.stats['processing_times']),
                'total_time': sum(self.stats['processing_times'])
            }
        
        return analysis
    
    def save_results(self, results: List[Dict], analysis: Dict):
        """保存测试结果"""
        # 保存详细结果
        results_path = os.path.join(self.output_dir, 'detection_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 保存分析结果
        analysis_path = os.path.join(self.output_dir, 'analysis.json')
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        # 生成报告
        report_path = os.path.join(self.output_dir, 'report.txt')
        with open(report_path, 'w') as f:
            f.write("Grounding DINO 在 MathVista 数据集上的测试报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据集路径: {self.data_path}\n")
            f.write(f"总样本数: {self.stats['total_samples']}\n")
            f.write(f"成功处理: {self.stats['processed_samples'] - self.stats['failed_samples']}\n")
            f.write(f"失败样本: {self.stats['failed_samples']}\n")
            f.write(f"总检测数: {self.stats['total_detections']}\n\n")
            
            f.write("任务类型性能:\n")
            f.write("-" * 60 + "\n")
            for task, stats in analysis['task_performance'].items():
                f.write(f"\n{task}:\n")
                f.write(f"  样本数: {stats['total_samples']}\n")
                f.write(f"  成功率: {stats['successful_samples']/stats['total_samples']*100:.1f}%\n")
                f.write(f"  平均检测数: {stats['avg_detections']:.1f}\n")
                f.write(f"  平均处理时间: {stats['avg_processing_time']:.3f}秒\n")
                f.write(f"  常见检测对象: {list(stats['common_detections'].keys())[:5]}\n")
            
            if 'processing_stats' in analysis:
                f.write("\n处理时间统计:\n")
                f.write("-" * 60 + "\n")
                f.write(f"平均时间: {analysis['processing_stats']['avg_time']:.3f}秒\n")
                f.write(f"最快时间: {analysis['processing_stats']['min_time']:.3f}秒\n")
                f.write(f"最慢时间: {analysis['processing_stats']['max_time']:.3f}秒\n")
                f.write(f"总耗时: {analysis['processing_stats']['total_time']:.1f}秒\n")
        
        print(f"\n结果已保存到: {self.output_dir}")
    
    def visualize_samples(self, results: List[Dict], num_samples: int = 5):
        """可视化一些检测结果"""
        vis_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # 选择有检测结果的样本
        samples_with_detections = [r for r in results if not r['errors'] and any(
            d['num_detections'] > 0 for d in r['detections'].values()
        )]
        
        # 可视化前N个样本
        for i, result in enumerate(samples_with_detections[:num_samples]):
            sample_id = result['id']
            
            # 找到对应的原始数据
            dataset = self.load_dataset()
            sample = next((s for s in dataset if s['id'] == sample_id), None)
            if not sample:
                continue
            
            # 加载图像
            self.tool.reset(sample['image_path'])
            
            # 选择检测结果最多的策略进行可视化
            best_detection = max(
                result['detections'].items(),
                key=lambda x: x[1]['num_detections']
            )
            
            strategy_name, detection_info = best_detection
            
            # 重新执行检测以获取完整结果
            detection_params = {'caption': detection_info['caption']}
            detection_result = self.tool.execute(json.dumps(detection_params))
            
            if 'error' not in detection_result:
                # 可视化
                vis_path = os.path.join(vis_dir, f"{sample_id}_{strategy_name}.jpg")
                self.tool.visualize_results(detection_result, save_path=vis_path)
                print(f"  可视化保存: {vis_path}")
    
    def run_test(self, max_samples: int = None, visualize: bool = True):
        """运行完整测试"""
        print("\n开始测试 Grounding DINO 在 MathVista 数据集上的性能\n")
        
        # 加载数据集
        dataset = self.load_dataset()
        if max_samples:
            dataset = dataset[:max_samples]
        
        self.stats['total_samples'] = len(dataset)
        
        # 处理每个样本
        results = []
        for sample in tqdm(dataset, desc="处理样本"):
            result = self.process_sample(sample)
            results.append(result)
            
            self.stats['processed_samples'] += 1
            if result['errors']:
                self.stats['failed_samples'] += 1
                print(f"\n样本 {result['id']} 处理失败:")
                for error in result['errors']:
                    print(f"  - {error}")
            
            # 更新任务统计
            task = result['task']
            if task not in self.stats['task_stats']:
                self.stats['task_stats'][task] = {'total': 0, 'success': 0}
            self.stats['task_stats'][task]['total'] += 1
            if not result['errors']:
                self.stats['task_stats'][task]['success'] += 1
        
        # 分析结果
        print("\n分析结果...")
        analysis = self.analyze_results(results)
        
        # 保存结果
        print("保存结果...")
        self.save_results(results, analysis)
        
        # 可视化一些样本
        if visualize:
            print("\n生成可视化...")
            self.visualize_samples(results, num_samples=10)
        
        # 打印摘要
        print("\n" + "="*60)
        print("测试完成！")
        print(f"总样本数: {self.stats['total_samples']}")
        print(f"处理样本数: {self.stats['processed_samples']}")
        print(f"成功处理: {self.stats['processed_samples'] - self.stats['failed_samples']}")
        print(f"失败样本: {self.stats['failed_samples']}")
        print(f"总检测数: {self.stats['total_detections']}")
        
        success_count = self.stats['processed_samples'] - self.stats['failed_samples']
        if success_count > 0:
            print(f"平均每样本检测数: {self.stats['total_detections'] / success_count:.1f}")
        else:
            print("平均每样本检测数: N/A (无成功样本)")
        
        if self.stats['processing_times']:
            avg_time = sum(self.stats['processing_times']) / len(self.stats['processing_times'])
            print(f"平均处理时间: {avg_time:.3f}秒")
        
        print(f"\n详细结果保存在: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="测试 Grounding DINO 在 MathVista 数据集上的性能")
    parser.add_argument('--data-path', type=str, 
                        default='/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/mathvista/mathvista_testmini.json',
                        help='MathVista 数据集路径')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='最大测试样本数')
    parser.add_argument('--no-visualize', action='store_true',
                        help='不生成可视化')
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = GroundingDINOTester(
        data_path=args.data_path,
        output_dir=args.output_dir
    )
    
    # 运行测试
    tester.run_test(
        max_samples=args.max_samples,
        visualize=not args.no_visualize
    )


if __name__ == "__main__":
    main()