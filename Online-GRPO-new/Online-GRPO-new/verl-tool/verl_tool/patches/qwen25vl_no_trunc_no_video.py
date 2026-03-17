# verl_tool/patches/qwen25vl_max_length_only.py
# -*- coding: utf-8 -*-
import os

from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor as _P

if not getattr(_P, "_no_trunc_patch", False):
    _orig_call = _P.__call__

    def _patched_call(self, *args, **kwargs):
        # 关键：禁截断 + 给足够大的 max_length（仅作为上限，不强制 padding）
        kwargs["truncation"] = False
        kwargs.setdefault("padding", "longest")
        # 不降低上游传入的更大值
        req = int(kwargs.get("max_length", 0) or 0)
        kwargs["max_length"] = max(req, 100000)

        out = _orig_call(self, *args, **kwargs)

        # 同步抬高 tokenizer 的上限，防止内层再回落到 2048
        try:
            tok = self.tokenizer
            if getattr(tok, "model_max_length", 0) < kwargs["max_length"]:
                tok.model_max_length = kwargs["max_length"]
            if isinstance(getattr(tok, "init_kwargs", None), dict):
                tok.init_kwargs["model_max_length"] = tok.model_max_length
        except Exception:
            pass

        if not getattr(_P, "_no_trunc_logged", False):
            print(f"[PATCH] Qwen2.5-VL truncation=False, max_length={kwargs['max_length']} (pid={os.getpid()})", flush=True)
            _P._no_trunc_logged = True
        return out

    _P.__call__ = _patched_call
    _P._no_trunc_patch = True

print("[PATCH] qwen25vl_max_length_only loaded", flush=True)
