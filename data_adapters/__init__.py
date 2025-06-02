
try:
    from .unified_adapter import UnifiedVQAAdapter
except ImportError:
    pass

try:
    from .chartqa_adapter import ChartQAAdapter
except ImportError:
    pass

try:
    from .scienceqa_adapter import ScienceQAAdapter
except ImportError:
    pass

__all__ = ["UnifiedVQAAdapter"]
