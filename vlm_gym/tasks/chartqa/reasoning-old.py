"""ChartQA reasoning engine - Complete version with all question types"""
from typing import Dict, Any, Optional, List
from .tools import ChartQATools
from .templates import ChartQATemplates
import re

class ChartQAReasoner:
    """Main reasoning engine for ChartQA tasks"""
    
    def __init__(self):
        self.tools = ChartQATools()
        self.templates = ChartQATemplates()
        self.debug = False  # 可以通过配置设置
    
    def classify_question(self, question: str) -> str:
        """Classify question type - comprehensive version"""
        question_lower = question.lower()
        
        # 1. Counting questions (how many)
        if any(phrase in question_lower for phrase in [
            'how many', 'count', 'total number', 'number of'
        ]) and not any(word in question_lower for word in ['sum', 'add', 'total of']):
            return 'counting'
        
        # 2. Summation questions
        elif any(phrase in question_lower for phrase in [
            'sum', 'add', 'total of', 'sum of', 'add up', 'combined'
        ]):
            return 'summation'
        
        # 3. Average questions
        elif any(word in question_lower for word in [
            'average', 'mean', 'avg'
        ]):
            return 'average'
        
        # 4. Percentage questions
        elif any(word in question_lower for word in [
            'percentage', 'percent', '%', 'proportion', 'share'
        ]):
            return 'percentage'
        
        # 5. Difference questions
        elif any(word in question_lower for word in [
            'difference', 'gap', 'subtract', 'minus', 'between'
        ]) and 'which' not in question_lower:
            return 'difference'
        
        # 6. Ratio questions
        elif any(word in question_lower for word in [
            'ratio', 'times', 'divide', 'divided by', 'per'
        ]):
            return 'ratio'
        
        # 7. Comparison questions
        elif any(phrase in question_lower for phrase in [
            'compare', 'which is greater', 'which is higher', 
            'which is lower', 'which is more', 'which is less',
            'vs', 'versus', 'bigger', 'smaller'
        ]):
            return 'comparison'
        
        # 8. Min/Max questions
        elif any(word in question_lower for word in [
            'maximum', 'minimum', 'highest', 'lowest', 
            'most', 'least', 'max', 'min', 'largest', 'smallest',
            'greatest', 'peak', 'bottom'
        ]):
            return 'minmax'
        
        # 9. Trend questions
        elif any(word in question_lower for word in [
            'trend', 'increase', 'decrease', 'change',
            'growth', 'decline', 'rise', 'fall', 'pattern'
        ]):
            return 'trend'
        
        # 10. Numerical value questions (general)
        elif any(phrase in question_lower for phrase in [
            'what is the value', 'how much', 'what value', 'what number'
        ]):
            return 'numerical'
        
        # 11. Retrieval questions (what, which, when, where, who)
        elif any(word in question_lower for word in [
            'what', 'which', 'when', 'where', 'who', 'whose'
        ]):
            return 'retrieval'
        
        # 12. Other
        else:
            return 'other'
    
    def reason(self, question: str, vlm_output: str, question_type: Optional[str] = None) -> Dict[str, Any]:
        """Apply structured reasoning based on question type"""
        
        if question_type is None:
            question_type = self.classify_question(question)
        
        # 根据问题类型调用相应的推理方法
        reasoning_methods = {
            'counting': self._reason_counting,
            'summation': self._reason_summation,
            'average': self._reason_average,
            'percentage': self._reason_percentage,
            'difference': self._reason_difference,
            'ratio': self._reason_ratio,
            'comparison': self._reason_comparison,
            'minmax': self._reason_minmax,
            'trend': self._reason_trend,
            'numerical': self._reason_numerical,
            'retrieval': self._reason_retrieval,
            'other': self._reason_other
        }
        
        method = reasoning_methods.get(question_type, self._reason_other)
        return method(question, vlm_output)
    
    def _reason_counting(self, question: str, vlm_output: str) -> Dict[str, Any]:
        """Handle counting questions"""
        return self._handle_counting_question(question, vlm_output)
    
    def _reason_summation(self, question: str, vlm_output: str) -> Dict[str, Any]:
        """Handle summation questions"""
        if self.debug:
            print(f"[DEBUG] Handling summation question: {question[:50]}...")
        
        # 提取所有数值
        values = self.tools.extractor.extract_numbers(vlm_output)
        
        if values:
            # 计算总和
            total = sum(v['value'] for v in values)
            answer = str(int(total) if total.is_integer() else round(total, 2))
            
            # 格式化计算过程
            calculation = f"Sum = {' + '.join(str(v['value']) for v in values)} = {answer}"
            
            # 创建推理说明
            reasoning = f"Extracted values:\n"
            reasoning += "\n".join([f"- {v['label']}: {v['value']}" for v in values])
            reasoning += f"\n{calculation}"
            
            return {
                'answer': answer,
                'reasoning': reasoning,
                'confidence': 0.85,
                'structured': True,
                'calculator_used': True
            }
        
        # 如果没有找到数值，尝试从文本中提取答案
        return self._extract_simple_answer(vlm_output, 'summation')
    
    def _reason_average(self, question: str, vlm_output: str) -> Dict[str, Any]:
        """Handle average questions"""
        if self.debug:
            print(f"[DEBUG] Handling average question: {question[:50]}...")
        
        # 提取所有数值
        values = self.tools.extractor.extract_numbers(vlm_output)
        
        if values:
            # 计算平均值
            avg = sum(v['value'] for v in values) / len(values)
            answer = str(round(avg, 2))
            
            # 格式化计算过程
            calculation = f"Average = ({' + '.join(str(v['value']) for v in values)}) / {len(values)} = {answer}"
            
            # 创建推理说明
            reasoning = f"Extracted values:\n"
            reasoning += "\n".join([f"- {v['label']}: {v['value']}" for v in values])
            reasoning += f"\n{calculation}"
            
            return {
                'answer': answer,
                'reasoning': reasoning,
                'confidence': 0.85,
                'structured': True,
                'calculator_used': True
            }
        
        return self._extract_simple_answer(vlm_output, 'average')
    
    def _reason_percentage(self, question: str, vlm_output: str) -> Dict[str, Any]:
        """Handle percentage questions"""
        if self.debug:
            print(f"[DEBUG] Handling percentage question: {question[:50]}...")
        
        # 查找百分比数值
        percentage_pattern = r'(\d+(?:\.\d+)?)\s*%'
        matches = re.findall(percentage_pattern, vlm_output)
        
        if matches:
            # 如果找到百分比，返回第一个
            answer = f"{matches[0]}%"
            return {
                'answer': answer,
                'reasoning': f"Extracted percentage value: {answer}",
                'confidence': 0.9,
                'structured': True,
                'calculator_used': False
            }
        
        # 如果需要计算百分比
        values = self.tools.extractor.extract_numbers(vlm_output)
        if len(values) >= 2:
            # 假设第一个是部分，第二个是整体
            part = values[0]['value']
            whole = values[1]['value']
            if whole != 0:
                percentage = (part / whole) * 100
                answer = f"{round(percentage, 2)}%"
                
                calculation = f"Percentage = ({part} / {whole}) × 100 = {answer}"
                return {
                    'answer': answer,
                    'reasoning': calculation,
                    'confidence': 0.8,
                    'structured': True,
                    'calculator_used': True
                }
        
        return self._extract_simple_answer(vlm_output, 'percentage')
    
    def _reason_difference(self, question: str, vlm_output: str) -> Dict[str, Any]:
        """Handle difference questions"""
        if self.debug:
            print(f"[DEBUG] Handling difference question: {question[:50]}...")
        
        # 提取数值
        values = self.tools.extractor.extract_numbers(vlm_output)
        
        if len(values) >= 2:
            # 计算差值（通常是较大值减较小值）
            val1, val2 = values[0]['value'], values[1]['value']
            diff = abs(val1 - val2)
            answer = str(int(diff) if diff.is_integer() else round(diff, 2))
            
            calculation = f"Difference = |{val1} - {val2}| = {answer}"
            
            return {
                'answer': answer,
                'reasoning': calculation,
                'confidence': 0.85,
                'structured': True,
                'calculator_used': True
            }
        
        return self._extract_simple_answer(vlm_output, 'difference')
    
    def _reason_ratio(self, question: str, vlm_output: str) -> Dict[str, Any]:
        """Handle ratio questions"""
        if self.debug:
            print(f"[DEBUG] Handling ratio question: {question[:50]}...")
        
        # 提取数值
        values = self.tools.extractor.extract_numbers(vlm_output)
        
        if len(values) >= 2:
            # 计算比率
            val1, val2 = values[0]['value'], values[1]['value']
            if val2 != 0:
                ratio = val1 / val2
                answer = str(round(ratio, 2))
                
                calculation = f"Ratio = {val1} / {val2} = {answer}"
                
                return {
                    'answer': answer,
                    'reasoning': calculation,
                    'confidence': 0.85,
                    'structured': True,
                    'calculator_used': True
                }
        
        return self._extract_simple_answer(vlm_output, 'ratio')
    
    def _reason_comparison(self, question: str, vlm_output: str) -> Dict[str, Any]:
        """Handle comparison questions"""
        # 1. 提取数值
        values = self.tools.extractor.extract_numbers(vlm_output)
        
        # 2. 计算差值
        if len(values) >= 2:
            diff = self.tools.calculate_difference(
                values[0]['value'], 
                values[1]['value']
            )
            
            # 3. 格式化答案
            template = self.templates.comparison_template()
            reasoning = template['format'].format(
                element_a=values[0]['label'],
                element_b=values[1]['label'],
                value_a=values[0]['value'],
                value_b=values[1]['value'],
                result=diff
            )
            
            # 4. 决定答案格式
            answer = str(int(diff) if diff.is_integer() else round(diff, 2))
            
            return {
                'answer': answer,
                'reasoning': reasoning,
                'confidence': 0.9,
                'structured': True,
                'calculator_used': True
            }
        
        return self._extract_simple_answer(vlm_output, 'comparison')
    
    def _reason_minmax(self, question: str, vlm_output: str) -> Dict[str, Any]:
        """Handle min/max questions"""
        # 1. 提取所有数值
        values = self.tools.extractor.extract_numbers(vlm_output)
        
        if values:
            # 2. 找出最小/最大值
            result = self.tools.find_min_max(values)
            
            # 3. 判断问题要求最小还是最大
            question_lower = question.lower()
            if any(word in question_lower for word in ['minimum', 'lowest', 'least', 'smallest']):
                target = result['min']
                answer_type = "Minimum"
            else:
                target = result['max']
                answer_type = "Maximum"
            
            # 4. 格式化推理过程
            values_list = "\n".join([f"- {v['label']}: {v['value']}" for v in values])
            template = self.templates.minmax_template()
            reasoning = template['format'].format(
                values_list=values_list,
                min_label=result['min']['label'],
                min_value=result['min']['value'],
                max_label=result['max']['label'],
                max_value=result['max']['value'],
                answer=f"{target['label']} ({target['value']})"
            )
            
            # 5. 如果问题只要数值或标签
            if 'value' in question_lower or 'number' in question_lower:
                answer = str(target['value'])
            else:
                answer = target['label']
            
            return {
                'answer': answer,
                'reasoning': reasoning,
                'confidence': 0.85,
                'structured': True,
                'structured_answer': target['value']  # 保存数值用于评估
            }
        
        return self._extract_simple_answer(vlm_output, 'minmax')
    
    def _reason_trend(self, question: str, vlm_output: str) -> Dict[str, Any]:
        """Handle trend questions"""
        if self.debug:
            print(f"[DEBUG] Handling trend question: {question[:50]}...")
        
        # 查找趋势相关的关键词
        trend_keywords = {
            'increasing': ['increasing', 'rising', 'growing', 'upward', 'going up'],
            'decreasing': ['decreasing', 'falling', 'declining', 'downward', 'going down'],
            'stable': ['stable', 'constant', 'unchanged', 'flat', 'steady']
        }
        
        vlm_lower = vlm_output.lower()
        
        for trend, keywords in trend_keywords.items():
            if any(keyword in vlm_lower for keyword in keywords):
                return {
                    'answer': trend,
                    'reasoning': f"Identified trend: {trend}",
                    'confidence': 0.8,
                    'structured': True,
                    'calculator_used': False
                }
        
        return self._extract_simple_answer(vlm_output, 'trend')
    
    def _reason_numerical(self, question: str, vlm_output: str) -> Dict[str, Any]:
        """Handle general numerical questions"""
        # 如果是 "how many" 类型，转到计数处理
        if 'how many' in question.lower():
            return self._handle_counting_question(question, vlm_output)
        
        # 提取数值
        values = self.tools.extractor.extract_numbers(vlm_output)
        
        if values:
            # 通常返回第一个找到的数值
            answer = str(values[0]['value'])
            
            return {
                'answer': answer,
                'reasoning': f"Extracted value: {answer}",
                'confidence': 0.8,
                'structured': True,
                'calculator_used': False
            }
        
        return self._extract_simple_answer(vlm_output, 'numerical')
    
    def _reason_retrieval(self, question: str, vlm_output: str) -> Dict[str, Any]:
        """Handle retrieval questions (what, which, when, where, who)"""
        if self.debug:
            print(f"[DEBUG] Handling retrieval question: {question[:50]}...")
        
        # 对于检索类问题，通常答案是一个标签、名称或描述
        # 尝试从VLM输出中提取简短的答案
        
        # 查找常见的答案模式
        answer_patterns = [
            r'(?:The answer is|Answer:)\s*([^.!?\n]+)',
            r'(?:It is|It\'s|This is)\s*([^.!?\n]+)',
            r'(?:^|\n)([A-Z][^.!?\n]{5,50})(?:\.|!|\?|$)',  # 以大写字母开头的短句
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, vlm_output, re.MULTILINE)
            if match:
                answer = match.group(1).strip()
                return {
                    'answer': answer,
                    'reasoning': f"Retrieved information: {answer}",
                    'confidence': 0.7,
                    'structured': True,
                    'calculator_used': False
                }
        
        # 如果没有找到特定模式，返回整个响应的第一句
        first_sentence = vlm_output.split('.')[0].strip()
        if len(first_sentence) < 100:  # 确保不是太长
            return {
                'answer': first_sentence,
                'reasoning': "Extracted from response",
                'confidence': 0.5,
                'structured': True,
                'calculator_used': False
            }
        
        return self._extract_simple_answer(vlm_output, 'retrieval')
    
    def _reason_other(self, question: str, vlm_output: str) -> Dict[str, Any]:
        """Handle other types of questions"""
        return self._extract_simple_answer(vlm_output, 'other')
    
    def _extract_simple_answer(self, vlm_output: str, question_type: str) -> Dict[str, Any]:
        """Extract simple answer when structured reasoning fails"""
        # 尝试提取最后的答案
        answer_patterns = [
            r'(?:The answer is|Answer:)\s*([^.!?\n]+)',
            r'(?:Therefore|Thus|So),?\s*([^.!?\n]+)',
            r'(?:^|\n)(\d+(?:\.\d+)?)',  # 独立的数字
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, vlm_output, re.MULTILINE | re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                return {
                    'answer': answer,
                    'reasoning': f"Extracted from VLM output ({question_type})",
                    'confidence': 0.4,
                    'structured': False,
                    'calculator_used': False
                }
        
        # 如果还是找不到，返回原始输出的前100个字符
        answer = vlm_output.strip()[:100]
        if len(vlm_output.strip()) > 100:
            answer += "..."
        
        return {
            'answer': answer,
            'confidence': 0.3,
            'structured': False,
            'calculator_used': False
        }
    
    def _handle_counting_question(self, question: str, vlm_output: str) -> Dict[str, Any]:
        """特殊处理计数类问题 - 增强版"""
        import re
        
        # 调试输出
        if self.debug:
            print(f"[DEBUG] Handling counting question: {question[:50]}...")
            print(f"[DEBUG] VLM output: {vlm_output[:200]}...")
        
        # 扩展的模式列表，按优先级排序
        patterns = [
            # 直接数字答案（在开头或独立行）
            (r'^(\d{1,2})$', 0.95),  # 1-2位数字在行首
            (r'^(\d{1,2})\.$', 0.95),  # 1-2位数字后跟句号
            
            # "X years/values/colors"等 - 限制为1-2位数
            (r'(\d{1,2})\s+(?:years?|values?|colors?|items?|points?|entries?|data\s*points?|countries?|bars?|lines?)', 0.9),
            (r'(?:compares?\s+data\s+for|uses?|has?|contains?|shows?|features?|represents?)\s+(\d{1,2})\s+(?:years?|values?|colors?|countries?)', 0.9),
            
            # "there are X"格式 - 限制为1-3位数
            (r'there\s+(?:are|is)\s+(\d{1,3})(?:\s|$)', 0.9),
            (r'(?:found|counted|identified)\s+(\d{1,3})(?:\s|$)', 0.9),
            
            # "X values are/were"格式
            (r'(\d{1,3})\s+(?:values?|points?|items?|countries?|bars?)\s+(?:are|were|is|was)', 0.85),
            
            # 包含数字的各种表达
            (r'(?:answer|total|count)(?:\s+is)?:?\s*(\d{1,3})(?:\s|$)', 0.85),
            (r'(?:exactly|total\s+of|in\s+total)\s+(\d{1,3})(?:\s|$)', 0.85),
            
            # 提取任何独立的数字（作为最后手段）- 但需要过滤
            (r'(?:^|\s)(\d+)(?:\s|$)', 0.5),
        ]
        
        # 数字单词映射
        word_to_num = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
            'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
            'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
            'eighteen': 18, 'nineteen': 19, 'twenty': 20
        }
        
        # 先检查数字单词
        for word, num in word_to_num.items():
            if word in vlm_output.lower():
                if self.debug:
                    print(f"[DEBUG] Found number word '{word}': {num}")
                return {
                    'answer': str(num),
                    'reasoning': f"Number word '{word}' extracted from text",
                    'confidence': 0.9,
                    'structured': True,
                    'calculator_used': False
                }
        
        # 存储所有找到的候选答案
        candidates = []
        
        # 尝试所有模式
        for i, (pattern, confidence_base) in enumerate(patterns):
            matches = re.findall(pattern, vlm_output, re.IGNORECASE | re.MULTILINE)
            if matches:
                for match in matches:
                    try:
                        num = int(match)
                        
                        # 过滤掉年份（1900-2099）除非问题明确是关于年份
                        if 1900 <= num <= 2099 and 'year' not in question.lower():
                            if self.debug:
                                print(f"[DEBUG] Filtered out year: {num}")
                            continue
                        
                        # 对于 "how many" 问题，大于100的数字可能性较低
                        if num > 100 and 'how many' in question.lower():
                            confidence = confidence_base * 0.5  # 降低置信度
                        else:
                            confidence = confidence_base
                        
                        candidates.append({
                            'value': num,
                            'pattern_index': i,
                            'confidence': confidence,
                            'pattern': pattern
                        })
                        
                    except ValueError:
                        continue
        
        # 根据置信度和合理性选择最佳候选
        if candidates:
            # 优先选择小数字（对于计数问题更合理）
            candidates.sort(key=lambda x: (
                -x['confidence'],  # 置信度高的优先
                x['value'] if x['value'] < 50 else 1000 + x['value']  # 小于50的数字优先
            ))
            
            best_candidate = candidates[0]
            if self.debug:
                print(f"[DEBUG] Selected count: {best_candidate['value']} (confidence: {best_candidate['confidence']:.2f})")
            
            return {
                'answer': str(best_candidate['value']),
                'reasoning': f"Count extracted with pattern {best_candidate['pattern_index']+1}",
                'confidence': best_candidate['confidence'],
                'structured': True,
                'calculator_used': False
            }
        
        # 如果没有找到明确的计数，返回未结构化的结果
        if self.debug:
            print(f"[DEBUG] Could not extract count, returning original output")
        
        return {
            'answer': vlm_output,
            'confidence': 0.3,
            'structured': False,
            'calculator_used': False
        }