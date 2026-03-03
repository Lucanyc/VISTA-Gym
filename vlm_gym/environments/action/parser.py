# vlm_gym/environments/action/parser.py
"""
这里是解析action动作的
"""
import ast
import re
import json
from typing import Dict, Any, Tuple, Optional

class ActionParser:
    """动作解析器"""
    
    def parse(self, action_str: str) -> Tuple[str, Dict[str, Any]]:
        """
        解析动作字符串
        
        支持格式:
        - action_name(param1=value1, param2=value2)  # 原有格式
        - action_name(value1, value2)                # 原有格式
        - action_name()                              # 原有格式
        - <tool_call>...</tool_call>                 # DeepEyes格式
        - <answer>...</answer>                       # DeepEyes格式
        """
        action_str = action_str.strip()
        
        # 首先检查是否是DeepEyes格式
        deepeyes_result = self._parse_deepeyes_format(action_str)
        if deepeyes_result:
            return deepeyes_result
        
        # 如果不是DeepEyes格式，使用原有的函数调用解析
        match = re.match(r'^(\w+)\s*\((.*)\)$', action_str)
        if not match:
            raise ValueError(f"Invalid action format: {action_str}")
        
        action_name = match.group(1)
        params_str = match.group(2).strip()
        
        if not params_str:
            return action_name, {}
        
        # 解析参数
        params = self._parse_params(params_str, action_name)
        return action_name, params
    
    def _parse_deepeyes_format(self, action_str: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        解析DeepEyes格式的action
        返回 (action_type, params) 或 None
        """
        # 提取think内容（可选）
        think_content = ""
        think_match = re.search(r'<think>(.*?)</think>', action_str, re.DOTALL)
        if think_match:
            think_content = think_match.group(1).strip()
        
        # 检查是否是工具调用
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
                # 如果JSON解析失败，返回错误信息
                return "deepeyes_tool_call_error", {
                    "error": f"Invalid JSON in tool_call: {str(e)}",
                    "raw_json": tool_json_str,
                    "think_content": think_content,
                    "raw_action": action_str
                }
        
        # 检查是否是答案
        answer_match = re.search(r'<answer>(.*?)</answer>', action_str, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            return "deepeyes_answer", {
                "answer": answer_content,
                "think_content": think_content,
                "raw_action": action_str
            }
        
        # 如果包含think但没有其他标签，可能是中间思考步骤
        if think_content:
            return "deepeyes_think_only", {
                "think_content": think_content,
                "raw_action": action_str
            }
        
        # 不是DeepEyes格式
        return None
    
    def parse_deepeyes_action(self, action_str: str) -> Tuple[str, Any]:
        """
        专门用于解析DeepEyes格式的便捷方法
        返回 (action_type, content)
        
        action_type可能的值:
        - "tool_call": 工具调用
        - "answer": 最终答案
        - "think": 仅思考
        - "text": 普通文本
        - "error": 解析错误
        """
        result = self._parse_deepeyes_format(action_str)
        
        if not result:
            # 不是DeepEyes格式，作为普通文本
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
    
    # 以下是原有的方法，保持不变
    def _parse_params(self, params_str: str, action_name: str) -> Dict[str, Any]:
        """解析参数"""
        try:
            # 使用AST安全解析
            code = f"{action_name}({params_str})"
            tree = ast.parse(code, mode='eval')
            
            if not isinstance(tree.body, ast.Call):
                raise ValueError("Not a valid function call")
            
            call_node = tree.body
            params = {}
            
            # 处理位置参数
            for i, arg in enumerate(call_node.args):
                value = self._eval_node(arg)
                params[f"arg{i}"] = value
            
            # 处理关键字参数
            for keyword in call_node.keywords:
                if keyword.arg is None:
                    raise ValueError("**kwargs not supported")
                value = self._eval_node(keyword.value)
                params[keyword.arg] = value
            
            return params
            
        except:
            # 降级到简单解析
            return self._simple_parse(params_str)
    
    def _eval_node(self, node: ast.AST) -> Any:
        """安全评估AST节点"""
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
        """简单参数解析备份"""
        params = {}
        
        # 尝试key=value格式
        pattern = r'(\w+)\s*=\s*([^,]+?)(?=\s*,\s*\w+\s*=|$)'
        matches = re.finditer(pattern, params_str)
        
        for match in matches:
            key = match.group(1)
            value_str = match.group(2).strip()
            params[key] = self._parse_value(value_str)
        
        return params
    
    def _parse_value(self, value_str: str) -> Any:
        """解析值"""
        value_str = value_str.strip()
        
        # 字符串
        if (value_str.startswith('"') and value_str.endswith('"')) or \
           (value_str.startswith("'") and value_str.endswith("'")):
            return value_str[1:-1]
        
        # 数字
        try:
            return float(value_str) if '.' in value_str else int(value_str)
        except ValueError:
            pass
        
        # 布尔值/None
        if value_str.lower() in ('true', 'false', 'none'):
            return eval(value_str.capitalize())
        
        # 列表
        if value_str.startswith('[') and value_str.endswith(']'):
            try:
                return ast.literal_eval(value_str)
            except:
                pass
        
        # 默认字符串
        return value_str


# 便捷函数，可以直接导入使用
def parse_deepeyes_action(action_string: str) -> Tuple[str, Any]:
    """
    解析DeepEyes格式的action
    返回 (action_type, content)
    """
    parser = ActionParser()
    return parser.parse_deepeyes_action(action_string)