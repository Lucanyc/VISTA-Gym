# vlm_gym/environments/action/action_set.py
"""
VLM Action Set Manager
"""
from typing import Dict, Any, Optional, List, Callable, Tuple
import inspect
from .base import AbstractActionSet
from . import function  # import from function.py
from . import vlm_actions  # added this line - import vlm_actions module
from .parser import ActionParser

class VLMActionSet(AbstractActionSet):
    """VLM Action Set Manager – responsible for registering, managing, and executing actions"""
    
    def __init__(self, custom_actions: Optional[Dict[str, Callable]] = None, strict: bool = False):
        super().__init__(strict=strict)
        
        # Initialize parser
        self.parser = ActionParser()
        
        # Automatically import all actions from function module
        self.action_set = {}
        self._load_core_actions()
        self._load_vlm_actions()  # added this line - load VLM actions
        
        # Add custom actions
        if custom_actions:
            for name, func in custom_actions.items():
                self._register_action(name, func)
    
    def _load_core_actions(self):
        """Load all core actions from function module"""
        # Get all functions inside function module
        for name, func in inspect.getmembers(function, inspect.isfunction):
            # Skip private functions
            if not name.startswith('_'):
                self._register_action(name, func)
    
    def _load_vlm_actions(self):  # this is the addition of new VLM actions
        """Load all VLM-specific actions from vlm_actions module"""
        # Get all functions inside vlm_actions module
        for name, func in inspect.getmembers(vlm_actions, inspect.isfunction):
            # Skip private functions
            if not name.startswith('_'):
                self._register_action(name, func)
    
    def _register_action(self, name: str, func: Callable):
        """Register an action"""
        # Parse function signature
        signature = inspect.signature(func)
        docstring = inspect.getdoc(func) or ""
        
        # Extract description and examples
        description, examples = self._parse_docstring(docstring)
        
        # Register action
        self.action_set[name] = {
            "function": func,
            "signature": str(signature),
            "description": description,
            "examples": examples,
            "params": list(signature.parameters.keys())
        }
    
    def _parse_docstring(self, docstring: str) -> Tuple[str, List[str]]:
        """Parse docstring to extract description and examples"""
        if not docstring:
            return "", []
        
        lines = docstring.strip().split('\n')
        description = []
        examples = []
        in_examples = False
        
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.lower() == "examples:":
                in_examples = True
                continue
            
            if in_examples:
                if line_stripped and not line.startswith('    '):
                    # Non-indented non-empty line ends example section
                    break
                if line_stripped:
                    examples.append(line_stripped)
            else:
                if line_stripped and not line_stripped.lower().startswith("examples:"):
                    description.append(line_stripped)
        
        return ' '.join(description), examples
    
    def list_actions(self) -> List[str]:
        """List all available actions"""
        return list(self.action_set.keys())
    
    def describe(self, with_long_description: bool = True, with_examples: bool = True) -> str:
        """Return a text description of the whole action space"""
        description = f"{len(self.action_set)} different types of actions are available.\n\n"
        
        # Sort alphabetically
        for action_name in sorted(self.action_set.keys()):
            action_info = self.action_set[action_name]
            
            # Action signature
            description += f"{action_name}{action_info['signature']}\n"
            
            # Long description
            if with_long_description and action_info['description']:
                description += f"    Description: {action_info['description']}\n"
            
            # Action examples
            if with_examples and action_info['examples']:
                description += "    Examples:\n"
                for example in action_info['examples']:
                    description += f"        {example}\n"
            
            description += "\n"
        
        return description
    
    def parse_action(self, action_str: str) -> Dict[str, Any]:
        """Parse action string"""
        try:
            action_name, params = self.parser.parse(action_str)
            
            # Validate action existence
            if action_name not in self.action_set:
                if self.strict:
                    raise ValueError(f"Unknown action: {action_name}. Available actions: {list(self.action_set.keys())}")
                else:
                    return {"action": action_name, "params": params, "valid": False}
            
            return {"action": action_name, "params": params, "valid": True}
        except Exception as e:
            if self.strict:
                raise
            return {"error": str(e), "valid": False}
    
    def execute_action(self, action_str: str) -> Dict[str, Any]:
        """Execute action and return result"""
        try:
            # Parse action
            parsed = self.parse_action(action_str)
            
            if not parsed.get("valid", True):
                return {
                    "type": "action_execution",
                    "status": "FAILED",
                    "error": f"Unknown action: {parsed.get('action', 'unknown')}",
                    "available_actions": self.list_actions()
                }
            
            action_name = parsed["action"]
            params = parsed["params"]
            
            # Get action function
            action_func = self.action_set[action_name]["function"]
            
            # Execute action
            result = action_func(**params)
            
            return {
                "type": "action_execution",
                "status": "SUCCESS",
                "action": action_name,
                "result": result
            }
            
        except Exception as e:
            error_msg = str(e)
            # Parse argument mismatch for better error message
            if "missing" in error_msg and "required positional argument" in error_msg:
                action_name = parsed.get("action", "unknown")
                if action_name in self.action_set:
                    required_params = self.action_set[action_name]["params"]
                    error_msg = f"Missing required parameters. Expected: {required_params}"
            
            return {
                "type": "action_execution",
                "status": "FAILED",
                "action": action_str,
                "error": error_msg
            }
    
    def validate_action(self, action_str: str) -> bool:
        """Validate whether the action is valid"""
        try:
            parsed = self.parse_action(action_str)
            return parsed.get("valid", False)
        except:
            return False
    
    def get_action_info(self, action_name: str) -> Dict[str, Any]:
        """Get detailed information of a specific action"""
        if action_name not in self.action_set:
            raise ValueError(f"Unknown action: {action_name}")
        
        return self.action_set[action_name].copy()
    
    def example_action(self, action_name: str) -> List[str]:
        """Return example actions"""
        if action_name not in self.action_set:
            raise ValueError(f"Unknown action: {action_name}")
        
        return self.action_set[action_name]["examples"]
