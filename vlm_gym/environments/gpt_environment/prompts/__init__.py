# vlm_gym/environments/gpt_environment/prompts/__init__.py

"""
Prompt templates for GPT-as-Environment
"""

from .task_prompts import TaskPrompts

__all__ = ['TaskPrompts']

# Convenience function
def get_prompts():
    """Get an instance of the prompts manager"""
    return TaskPrompts()