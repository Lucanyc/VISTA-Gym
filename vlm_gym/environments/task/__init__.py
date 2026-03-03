"""
VLM-Gym 任务模块
包含所有任务相关的类和功能
"""

# 导入基类
from .base import BaseTask, BaseAdapter

# 导入具体任务
from .vision_qa_task import VisionQATask

# 导入其他任务
try:
    from .chartqa import ChartQATask
except ImportError:
    ChartQATask = None

try:
    from .scienceqa import ScienceQATask
except ImportError:
    ScienceQATask = None

try:
    from .geometry3k import Geometry3KTask  # 修正：geometry3k 不是 geometry2k
except ImportError:
    Geometry3KTask = None  # 修正变量名

try:
    from .mathvista import MathVistaTask
except ImportError:
    MathVistaTask = None


try:
    from .olympiadbench import OlympiadBenchTask
except ImportError:
    OlympiadBenchTask = None

try:
    from .geoqa import GeoQATask
except ImportError:
    GeoQATask = None

try:
    from .clevr import CLEVRTask
except ImportError:
    CLEVRTask = None
    

try:
    from .iconqa import IconQATask
except ImportError:
    IconQATask = None


try:
    from .figureqa import FigureQATask
except ImportError:
    FigureQATask = None


try:
    from .tabmwp import TabMWPTask
except ImportError:
    TabMWPTask = None


try:
    from .ai2d import AI2DTask
except ImportError:
    AI2DTask = None


try:
    from .plotqa import PlotQATask
except ImportError:
    PlotQATask = None



try:
    from .mapqa import MapQATask
except ImportError:
    MapQATask = None


try:
    from .vizwiz import VizWizTask
except ImportError:
    VizWizTask = None


try:
    from .geos import GEOSTask
except ImportError:
    GEOSTask = None


try:
    from .docvqa import DocVQATask
except ImportError:
    DocVQATask = None


try:
    from .textvqa import TextVQATask
except ImportError:
    TextVQATask = None



try:
    from .clevermath import ClevrMathTask
except ImportError:
    ClevrMathTask = None


try:
    from .vqacoco import VQACOCOTask
except ImportError:
    VQACOCOTask = None



try:
    from .superclevr import SuperClevrTask
except ImportError:
    SuperClevrTask = None


try:
    from .vqaas import VqaAsTask
except ImportError:
    VqaAsTask = None

try:
    from .vqarad import VqaRadTask
except ImportError:
    VqaAdapter = None


try:
    from .aokvqa import AOKVQATask
except ImportError:
    AOKVQATask = None


try:
    from .unigeo import UniGeoTask
except ImportError:
    UniGeoTask = None

# 导出
__all__ = [
    # 基类
    "BaseTask",
    "BaseAdapter",
    
    # 具体任务
    "VisionQATask",
]

# 只导出存在的任务
if ChartQATask is not None:
    __all__.append("ChartQATask")
    
if ScienceQATask is not None:
    __all__.append("ScienceQATask")
    
if Geometry3KTask is not None:  
    __all__.append("Geometry3KTask") 
    
if MathVistaTask is not None:
    __all__.append("MathVistaTask")
    
    
if OlympiadBenchTask is not None:
    __all__.append("OlympiadBenchTask")
    

if GeoQATask is not None:
    __all__.append("GeoQATask")
    
    
if CLEVRTask is not None:
    __all__.append("CLEVRTask")


if IconQATask is not None:
    __all__.append("IconQATask")
    

if FigureQATask is not None:
    __all__.append("FigureQATask")
    

if TabMWPTask is not None:
    __all__.append("TabMWPTask")
    
    
if AI2DTask is not None:
    __all__.append("AI2DTask")
    
    
    
if PlotQATask is not None:
    __all__.append("PlotQATask")
    

if MapQATask is not None:
    __all__.append("MapQATask")
    

if VizWizTask is not None:
    __all__.append("VizWizTask")
    
    
if GEOSTask is not None:
    __all__.append("GEOSTask")
    

if DocVQATask is not None:
    __all__.append("DocVQATask")
    
    

if TextVQATask is not None:
    __all__.append("TextQATask")
    

if ClevrMathTask is not None:
    __all__.append("CLEVRMathTask")
    

if VQACOCOTask is not None:
    __all__.append("VQACOCOTask")
    

if SuperClevrTask is not None:
    __all__.append("SuperClevrTask")
    
if VqaAsTask is not None:
    __all__.append("VqaAsTask")
    

if VqaRadTask is not None:
    __all__.append("VqaRadTask")
    
    
if AOKVQATask is not None:
    __all__.append("AOKVQATask")
    
if UniGeoTask is not None:
    __all__.append("UniGeoTask")