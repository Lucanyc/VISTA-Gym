# vlm_gym/environments/gpt_environment/exceptions.py

class GPTAPIError(Exception):
    """GPT API调用相关的错误"""
    pass

class PromptGenerationError(Exception):
    """提示生成相关的错误"""
    pass

class TaskLoadError(Exception):
    """任务加载相关的错误"""
    pass

class ConfigurationError(Exception):
    """配置相关的错误"""
    pass