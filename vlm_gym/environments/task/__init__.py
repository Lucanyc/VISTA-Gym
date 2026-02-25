
from .base import BaseTask, BaseAdapter

from .vision_qa_task import VisionQATask

try:
    from .chartqa import ChartQATask  
except ImportError:
    ChartQATask = None

try:
    from .scienceqa import ScienceQATask 
except ImportError:
    ScienceQATask = None


__all__ = [

    "BaseTask",
    "BaseAdapter",
    

    "VisionQATask",
]


if ChartQATask is not None:
    __all__.append("ChartQATask")
if ScienceQATask is not None:
    __all__.append("ScienceQATask")
