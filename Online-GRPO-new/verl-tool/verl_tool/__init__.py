# verl_tool/__init__.py
import os
if os.getenv("VERL_APPLY_QWEN25VL_PATCH", "1") == "1":
    import verl_tool.patches.qwen25vl_no_trunc_no_video  # noqa: F401
