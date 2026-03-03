# vlm_gym/environments/action/parser.py
"""
This module parses action strings
"""
import ast
import re
import json
from typing import Dict, Any, Tuple, Optional

class ActionParser:
    """Action parser"""
    
    def parse(self, action_str: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parse an action string
        
        Supported formats:
        - action_name(param1=value1, param2=value2)    # original format
        - action_name(value1, value2)                  # original format
        - action_name()                                # original format
        - <tool_call>...</tool_call>                   # DeepEyes format
        - <answer>...</answer>                         # DeepEyes format
        """
        action_str = action_str.strip()
        
        # First check whether it is the DeepEyes format
        deepeyes_result = self._parse_deepeyes_format(action_str)
        if deepeyes_result:
            return deepeyes_result
        
        # If not DeepEyes format, parse using the normal function call syntax
        match = re.match(r'^(\w+)\s*\((.*)\)$', action_str)
        if not match:
            raise ValueError(f"Invalid action format: {action_str}")
        
        action_name = match.group(1)
        params_str = match.group(2).strip()
        
        if not params_str:
            return action_name, {}
        
        # Parse parameters
        params = self._parse_params(params_str, action_name)
        return action_name, params
    
    def _parse_deepeyes_format(self, action_str: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Parse a DeepEyes-format action
        Returns (action_type, params) or None
        """
        # Extract think content (optional)
        think_content = ""
        think_match = re.search(r'<think>(.*?)</think>', action_str, re.DOTALL)
        if think_match:
            think_content = think_match.group(1).strip()
        
        # Check whether it is a tool call
        tool_call_match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', action_str, re.DOTALL)
        if tool_call_match:
            tool_json_str = tool_call_match.group(1).strip()
            try:
                tool_data = json.loads(tool_json_str)
                return "deepeyes_tool_call", {
                    "tool_name": tool_data.get("name", ""),
                    "arguments": tool_data.get("arguments", {}),
                    "think_content": think_content,
                    "raw_action": action_str
                }
            except json.JSONDecodeError as e:
                # If JSON parsing fails, return error information
                return "deepeyes_tool_call_error", {
                    "error": f"Invalid JSON in tool_call: {str(e)}",
                    "raw_json": tool_json_str,
                    "think_content": think_content,
                    "raw_action": action_str
                }
        
        # Check whether it is an answer
        answer_match = re.search(r'<answer>(.*?)</answer>', action_str, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            return "deepeyes_answer", {
                "answer": answer_content,
                "think_content": think_content,
                "raw_action": action_str
            }
        
        # If only think content exists but without other tags, treat it as intermediate thought
        if think_content:
            return "deepeyes_think_only", {
                "think_content": think_content,
                "raw_action": action_str
            }
        
        # Not DeepEyes format
        return None
    
    def parse_deepeyes_action(self, action_str: str) -> Tuple[str, Any]:
        """
        Convenience method for parsing DeepEyes-format actions
        Returns (action_type, content)
        
        possible action_type values:
        - "tool_call": tool invocation
        - "answer": final answer
        - "think": only thinking content
        - "text": normal text
        - "error": parsing error
        """
        result = self._parse_deepeyes_format(action_str)
        
        if not result:
            # Not DeepEyes format → treat as normal text
            return "text", action_str
        
        action_type, params = result
        
        if action_type == "deepeyes_tool_call":
            return "tool_call", params
        elif action_type == "deepeyes_tool_call_error":
            return "error", params
        elif action_type == "deepeyes_answer":
            return "answer", params["answer"]
        elif action_type == "deepeyes_think_only":
            return "think", params["think_content"]
        else:
            return "text", action_str
    
    # Below are original methods, unchanged
    def _parse_params(self, params_str: str, action_name: str) -> Dict[str, Any]:
        """Parse parameters"""
        try:
            # Use AST for safe parsing
            code = f"{action_name}({params_str})"
            tree = ast.parse(code, mode='eval')
            
            if not isinstance(tree.body, ast.Call):
                raise ValueError("Not a valid function call")
            
            call_node = tree.body
            params = {}
            
            # Positional arguments
            for i, arg in enumerate(call_node.args):
                value = self._eval_node(arg)
                params[f"arg{i}"] = value
            
            # Keyword arguments
            for keyword in call_node.keywords:
                if keyword.arg is None:
                    raise ValueError("**kwargs not supported")
                value = self._eval_node(keyword.value)
                params[keyword.arg] = value
            
            return params
            
        except:
            # Fallback to simple parsing
            return self._simple_parse(params_str)
    
    def _eval_node(self, node: ast.AST) -> Any:
        """Safely evaluate AST node"""
        if isinstance(node, (ast.Constant, ast.Num, ast.Str)):
            return node.n if hasattr(node, 'n') else (node.s if hasattr(node, 's') else node.value)
        elif isinstance(node, ast.List):
            return [self._eval_node(item) for item in node.elts]
        elif isinstance(node, ast.Dict):
            return {
                self._eval_node(k): self._eval_node(v)
                for k, v in zip(node.keys, node.values)
            }
        elif isinstance(node, ast.Name):
            if node.id in ('True', 'False', 'None'):
                return eval(node.id)
            return node.id
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self._eval_node(node.operand)
        else:
            raise ValueError(f"Unsupported node type: {type(node)}")
    
    def _simple_parse(self, params_str: str) -> Dict[str, Any]:
        """Simple backup parameter parser"""
        params = {}
        
        # Try key=value parsing
        pattern = r'(\w+)\s*=\s*([^,]+?)(?=\s*,\s*\w+\s*=|$)'
        matches = re.finditer(pattern, params_str)
        
        for match in matches:
            key = match.group(1)
            value_str = match.group(2).strip()
            params[key] = self._parse_value(value_str)
        
        return params
    
    def _parse_value(self, value_str: str) -> Any:
        """Parse parameter value"""
        value_str = value_str.strip()
        
        # String
        if (value_str.startswith('"') and value_str.endswith('"')) or \
           (value_str.startswith("'") and value_str.endswith("'")):
            return value_str[1:-1]
        
        # Number
        try:
            return float(value_str) if '.' in value_str else int(value_str)
        except ValueError:
            pass
        
        # Boolean / None
        if value_str.lower() in ('true', 'false', 'none'):
            return eval(value_str.capitalize())
        
        # List
        if value_str.startswith('[') and value_str.endswith(']'):
            try:
                return ast.literal_eval(value_str)
            except:
                pass
        
        # Default fallback: string
        return value_str


# Convenience function for direct import and usage
def parse_deepeyes_action(action_string: str) -> Tuple[str, Any]:
    """
    Parse a DeepEyes-format action
    Returns (action_type, content)
    """
    parser = ActionParser()
    return parser.parse_deepeyes_action(action_string)
