# vlm_gym/environments/strategies/figure_qa.py

class FigureQAStrategy:
    """图表问答任务的教学策略"""
    
    def __init__(self):
        self.analysis_framework = {
            "chart_type": "What type of chart is this?",
            "axes_labels": "What do the axes represent?",
            "data_points": "Can you identify the key data points?",
            "comparison": "How would you compare these values?",
            "conclusion": "Based on your analysis, what's the answer?"
        }