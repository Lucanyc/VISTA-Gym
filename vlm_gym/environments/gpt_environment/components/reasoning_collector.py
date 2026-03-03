# vlm_gym/environments/components/reasoning_collector.py

class ReasoningPathCollector:
    """收集和评估推理路径"""
    
    def __init__(self, quality_threshold=0.7):
        self.collected_paths = []
        self.quality_threshold = quality_threshold
        self.path_analyzer = PathAnalyzer()
    
    def collect_path(self, dialogue, task, outcome):
        """收集一条推理路径"""
        path = {
            'task_id': task.id,
            'task_type': task.type,
            'dialogue': dialogue,
            'reasoning_steps': self.extract_steps(dialogue),
            'quality_metrics': self.assess_quality(dialogue),
            'outcome': outcome,
            'timestamp': datetime.now()
        }
        
        if path['quality_metrics']['overall'] >= self.quality_threshold:
            self.collected_paths.append(path)
            self.save_to_database(path)
    
    def extract_steps(self, dialogue):
        """从对话中提取推理步骤"""
        steps = []
        for turn in dialogue:
            if turn['role'] == 'agent':
                step = self.path_analyzer.extract_reasoning(turn['content'])
                if step:
                    steps.append(step)
        return steps
    
    def assess_quality(self, dialogue):
        """评估推理质量"""
        return {
            'clarity': self.assess_clarity(dialogue),
            'completeness': self.assess_completeness(dialogue),
            'correctness': self.assess_correctness(dialogue),
            'depth': self.assess_depth(dialogue),
            'overall': self.compute_overall_score()
        }