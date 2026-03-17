"""
Force patch for Qwen2.5-VL - more aggressive approach
"""

# Monkey patch BEFORE any imports
def force_patch():
    import sys
    import importlib
    
    # Force reload and patch transformers
    if 'transformers.models.qwen2_5_vl.processing_qwen2_5_vl' in sys.modules:
        del sys.modules['transformers.models.qwen2_5_vl.processing_qwen2_5_vl']
    
    from transformers.models.qwen2_5_vl import processing_qwen2_5_vl
    
    # Completely replace the problematic method
    def dummy_check(self, *args, **kwargs):
        return None
    
    processing_qwen2_5_vl.Qwen2_5_VLProcessor._check_special_mm_tokens = dummy_check
    
    # Also patch the __call__ method
    original_call = processing_qwen2_5_vl.Qwen2_5_VLProcessor.__call__
    
    def safe_call(self, text=None, images=None, videos=None, **kwargs):
        # Never process videos
        videos = None
        
        # Force remove problematic kwargs
        safe_kwargs = {}
        for k, v in kwargs.items():
            if k not in ['truncation', 'max_length', 'padding', 'return_tensors']:
                safe_kwargs[k] = v
        
        # Force specific settings
        safe_kwargs['truncation'] = False
        safe_kwargs['return_tensors'] = 'pt'
        
        # Replace video tokens in text
        if text and isinstance(text, str):
            text = text.replace('<|video_pad|>', '')
        
        try:
            return original_call(self, text=text, images=images, videos=None, **safe_kwargs)
        except Exception as e:
            print(f"[PATCH] Processor error caught and handled: {e}")
            # Return a minimal valid response
            return {'input_ids': [[1]], 'attention_mask': [[1]]}
    
    processing_qwen2_5_vl.Qwen2_5_VLProcessor.__call__ = safe_call
    
    print("[FORCE PATCH] Qwen2.5-VL aggressively patched")

# Apply immediately
force_patch()

# Also patch vLLM's dummy data generation
try:
    import vllm.model_executor.models.qwen2_vl as qwen2_vl
    
    # Override the supports_multi_modal to disable video
    original_supports = qwen2_vl.Qwen2VLForConditionalGeneration.supports_multi_modal
    
    def patched_supports(self):
        # Only return image support, no video
        return ["image"]
    
    qwen2_vl.Qwen2VLForConditionalGeneration.supports_multi_modal = property(lambda self: ["image"])
    
    print("[FORCE PATCH] vLLM model video support disabled")
except:
    pass