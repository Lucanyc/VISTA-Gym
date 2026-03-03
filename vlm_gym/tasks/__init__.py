#!/usr/bin/env python3
"""VLM Gym Tasks Package"""

# Import base classes
from .base.task_wrapper import BaseTaskWrapper
from .base.components import (
    Evaluator,
    AnswerExtractor,
    QuestionClassifier,
    FeedbackGenerator,
    OutputFormatter,
    HintGenerator
)
from .base.registry import (
    TaskRegistry,
    register_task,
    get_task_class,
    create_task
)

# Import implementations (this will auto-register them)
try:
    from .implementations import chartqa
    CHARTQA_AVAILABLE = True
except ImportError:
    CHARTQA_AVAILABLE = False

# Try to import other task implementations
AVAILABLE_TASKS = []

if CHARTQA_AVAILABLE:
    AVAILABLE_TASKS.append("chartqa")

# Future task imports
task_modules = [
    ("figureqa", "implementations.figureqa"),
    ("scienceqa", "implementations.scienceqa"),
    ("geoqa", "implementations.geoqa"),
    ("plotqa", "implementations.plotqa"),
    ("ai2d", "implementations.ai2d"),
    ("vqarad", "implementations.vqarad"),
    ("geometry3k", "implementations.geometry3k"),
    ("mmmu", "implementations.mmmu")
]

for task_name, module_path in task_modules:
    try:
        exec(f"from .{module_path} import *")
        AVAILABLE_TASKS.append(task_name)
    except ImportError:
        pass

# Export public API
__all__ = [
    # Base classes
    "BaseTaskWrapper",
    "Evaluator",
    "AnswerExtractor",
    "QuestionClassifier",
    "FeedbackGenerator",
    "OutputFormatter",
    "HintGenerator",
    
    # Registry functions
    "TaskRegistry",
    "register_task",
    "get_task_class",
    "create_task",
    
    # Available tasks
    "AVAILABLE_TASKS"
]

print(f"VLM Tasks module loaded. Available tasks: {AVAILABLE_TASKS}")