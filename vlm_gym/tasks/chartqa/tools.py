"""ChartQA specific tools for numerical reasoning"""
import re
from typing import List, Dict, Any, Optional
import ast
import operator

class ChartQACalculator:
    """Safe calculator for ChartQA tasks"""
    
    # 安全的操作符
    SAFE_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
    }
    
    def calculate(self, expression: str) -> Optional[float]:
        """Safely evaluate mathematical expressions"""
        try:
            # 解析表达式
            node = ast.parse(expression, mode='eval')
            return self._eval_node(node.body)
        except:
            return None
    
    def _eval_node(self, node):
        """递归计算AST节点"""
        if isinstance(node, ast.Num):  # Python 3.7 及以下
            return node.n
        elif isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.BinOp):
            op = self.SAFE_OPERATORS.get(type(node.op))
            if op:
                return op(self._eval_node(node.left), self._eval_node(node.right))
        elif isinstance(node, ast.UnaryOp):
            op = self.SAFE_OPERATORS.get(type(node.op))
            if op:
                return op(self._eval_node(node.operand))
        raise ValueError(f"Unsupported operation: {node}")

class ChartQAValueExtractor:
    """Extract numerical values from text and charts"""
    
    @staticmethod
    def extract_numbers(text: str, filter_years: bool = True) -> List[Dict[str, Any]]:
        """
        Extract numbers with context - improved version with year filtering
        
        Args:
            text: Input text to extract numbers from
            filter_years: Whether to filter out year-like numbers (default: True)
        """
        results = []
        seen_values = set()  # 避免重复
        
        # 预处理文本
        text = text.replace('\n', ' ')
        
        # 多种模式匹配数字 - 改进版，优先匹配特定模式
        patterns = [
            # Pattern 1: "In YYYY, the X was VALUE" - 只提取VALUE
            (r'[Ii]n\s+(\d{4})[,:]?\s+(?:the\s+)?(?:\w+\s+)?(?:was|is|were|are)\s+(\d+(?:\.\d+)?)\s*(%?)', 'year_context'),
            
            # Pattern 2: "value in YYYY was VALUE" - 只提取VALUE
            (r'(?:value|rating|score)\s+in\s+(\d{4})\s+(?:was|is)\s+(\d+(?:\.\d+)?)\s*(%?)', 'value_in_year'),
            
            # Pattern 3: 标签: 数值 格式（但要过滤年份）
            (r'([A-Za-z][A-Za-z\s]+?):\s*(\d+(?:\.\d+)?)\s*(%?)', 'label_value'),
            
            # Pattern 4: 数值% 标签 格式
            (r'(\d+(?:\.\d+)?)\s*(%?)\s+([A-Za-z][A-Za-z\s]+)', 'value_label'),
            
            # Pattern 5: "X values/items" 格式（用于计数）
            (r'(\d+)\s+(values?|items?|entries?|points?|data\s*points?)', 'count_items'),
            
            # Pattern 6: "sum is VALUE" 或 "total is VALUE"
            (r'(?:sum|total)\s+is\s+(\d+(?:\.\d+)?)', 'calculated_sum'),
        ]
        
        # 先收集所有潜在的年份，用于后续过滤
        year_pattern = r'\b(19\d{2}|20\d{2})\b'
        potential_years = set(re.findall(year_pattern, text))
        
        # 标记已处理的位置，避免重复提取
        processed_positions = set()
        
        for pattern_info in patterns:
            pattern, pattern_type = pattern_info
            
            for match in re.finditer(pattern, text, re.IGNORECASE):
                match_start = match.start()
                match_end = match.end()
                
                # 检查是否已处理过这个位置的数字
                if any(start <= match_start <= end or start <= match_end <= end 
                       for start, end in processed_positions):
                    continue
                
                if pattern_type == 'year_context':
                    # "In 2014, the value was 51" - 只提取51
                    year = match.group(1)
                    value = match.group(2)
                    percent = match.group(3)
                    
                    num_value = float(value)
                    label = f"value_in_{year}"
                    display_value = f"{value}%" if percent else value
                    
                    # 标记年份位置为已处理
                    year_pos = text.find(year, match_start)
                    if year_pos != -1:
                        processed_positions.add((year_pos, year_pos + len(year)))
                    
                elif pattern_type == 'value_in_year':
                    # Similar to above
                    year = match.group(1)
                    value = match.group(2)
                    percent = match.group(3)
                    
                    num_value = float(value)
                    label = f"value_in_{year}"
                    display_value = f"{value}%" if percent else value
                    
                elif pattern_type == 'calculated_sum':
                    # "sum is 95" - 直接提取计算好的和
                    value = match.group(1)
                    num_value = float(value)
                    label = "calculated_sum"
                    display_value = value
                    
                elif pattern_type == 'label_value':
                    label = match.group(1)
                    value = match.group(2)
                    percent = match.group(3)
                    num_value = float(value)
                    
                    # 检查这个数值是否是年份
                    if filter_years and 1900 <= num_value <= 2099 and len(value) == 4:
                        # 检查标签是否暗示这不是年份
                        if not any(word in label.lower() for word in 
                                 ['percent', 'rating', 'value', 'score', 'favorable', 'unfavorable']):
                            continue
                    
                    display_value = f"{value}%" if percent else value
                    
                elif pattern_type == 'value_label':
                    value = match.group(1)
                    percent = match.group(2)
                    label = match.group(3)
                    num_value = float(value)
                    
                    # 过滤年份
                    if filter_years and 1900 <= num_value <= 2099 and len(value) == 4:
                        continue
                    
                    display_value = f"{value}%" if percent else value
                    
                elif pattern_type == 'count_items':
                    value = match.group(1)
                    unit = match.group(2)
                    label = unit
                    num_value = float(value)
                    display_value = value
                
                # 记录已处理的位置
                processed_positions.add((match_start, match_end))
                
                # 避免重复添加相同的值
                value_key = (label.strip().lower(), num_value)
                if value_key not in seen_values:
                    seen_values.add(value_key)
                    results.append({
                        'label': label.strip(),
                        'value': num_value,
                        'original': display_value
                    })
        
        # 如果没找到带标签的，尝试找独立数字（但要更严格的过滤）
        if not results:
            # 更精确的独立数字匹配
            number_pattern = r'(?<![a-zA-Z0-9])(\d+(?:\.\d+)?)(?![a-zA-Z0-9])'
            
            for match in re.finditer(number_pattern, text):
                num = match.group(1)
                match_pos = match.start()
                
                # 检查是否已处理
                if any(start <= match_pos <= end for start, end in processed_positions):
                    continue
                
                try:
                    num_value = float(num)
                    
                    # 严格的年份过滤
                    if filter_years and 1900 <= num_value <= 2099 and len(num) == 4:
                        # 检查前后文
                        before_start = max(0, match_pos - 30)
                        after_end = min(len(text), match_pos + len(num) + 30)
                        context = text[before_start:after_end].lower()
                        
                        # 年份指示词
                        year_indicators = ['year', 'in', 'during', 'from', 'to', 'between', 'since']
                        # 值指示词
                        value_indicators = ['value', 'rating', 'percent', 'score', '%', 'favorable', 'sum', 'total']
                        
                        # 如果上下文包含年份指示词，跳过
                        if any(ind in context for ind in year_indicators) and \
                           not any(val_ind in context for val_ind in value_indicators):
                            continue
                    
                    results.append({
                        'label': f'value_{len(results)+1}',
                        'value': num_value,
                        'original': num
                    })
                    
                except ValueError:
                    continue
        
        return results

class ChartQATools:
    """Collection of ChartQA specific tools"""
    
    def __init__(self):
        self.calculator = ChartQACalculator()
        self.extractor = ChartQAValueExtractor()
    
    def find_min_max(self, values: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find minimum and maximum values with labels"""
        if not values:
            return {'min': None, 'max': None}
        
        min_item = min(values, key=lambda x: x['value'])
        max_item = max(values, key=lambda x: x['value'])
        
        return {
            'min': min_item,
            'max': max_item,
            'range': max_item['value'] - min_item['value']
        }
    
    def calculate_difference(self, val1: float, val2: float) -> float:
        """Calculate difference between two values"""
        return abs(val1 - val2)
    
    def calculate_percentage_change(self, old_val: float, new_val: float) -> float:
        """Calculate percentage change"""
        if old_val == 0:
            return 0
        return ((new_val - old_val) / old_val) * 100
    
    def count_values_meeting_condition(self, values: List[Dict[str, Any]], condition: str, threshold: float) -> int:
        """Count values meeting a specific condition"""
        count = 0
        for v in values:
            if condition == 'below' and v['value'] < threshold:
                count += 1
            elif condition == 'above' and v['value'] > threshold:
                count += 1
            elif condition == 'equal' and v['value'] == threshold:
                count += 1
        return count
    
    def extract_counting_answer(self, text: str) -> Optional[int]:
        """Extract counting answer from text"""
        # 尝试多种模式
        patterns = [
            r'there\s+are\s+(\d+)',
            r'found\s+(\d+)',
            r'count(?:ed)?:?\s*(\d+)',
            r'total:?\s*(\d+)',
            r'answer:?\s*(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return None