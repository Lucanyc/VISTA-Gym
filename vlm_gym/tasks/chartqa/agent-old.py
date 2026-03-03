import re
import torch
from typing import Dict, Any, Tuple
from PIL import Image
from vlm_gym.agents.vlm_agent import VLMAgent
from .reasoning import ChartQAReasoner


class ChartQAAgent(VLMAgent):
    """VLM Agent enhanced with ChartQA reasoning and direct generation"""
    
    def __init__(self, config: Dict[str, Any]):
        # Extract ChartQA-specific parameters before passing to parent
        chartqa_params = {}
        chartqa_keys = ['enable_structured_reasoning', 'use_calculator', 'debug', '_chartqa_params']
        
        # If config has _chartqa_params, use it directly
        if '_chartqa_params' in config:
            chartqa_params = config.pop('_chartqa_params')
        else:
            # Otherwise extract ChartQA specific params
            for key in chartqa_keys:
                if key in config:
                    chartqa_params[key] = config.pop(key)
        
        # Ensure config has the expected structure with 'agent' key
        if 'agent' not in config:
            # Wrap the config in the expected structure
            vlm_config = {"agent": config}
        else:
            vlm_config = config
            
        # Call parent class initialization with cleaned config
        super().__init__(vlm_config)
        
        # Initialize ChartQA specific components
        self.chartqa_reasoner = ChartQAReasoner()
        
        # ChartQA specific configuration - now from extracted params
        self.enable_structured_reasoning = chartqa_params.get('enable_structured_reasoning', True)
        self.use_calculator = chartqa_params.get('use_calculator', True)
        self.debug = chartqa_params.get('debug', False)
        
        # Ensure model is loaded
        self.load_model()
        
        print(f"[ChartQAAgent] Initialized with debug={self.debug}, structured_reasoning={self.enable_structured_reasoning}")
    
    def load_model(self):
        """强制加载模型"""
        if not hasattr(self, '_loaded') or not self._loaded:
            # 调用父类的 load_model 方法
            super().load_model()
    
    def _clean_model_output(self, output: str) -> str:
        """清理模型输出中的异常字符和模式"""
        if not output:
            return output
            
        original_output = output  # 保存原始输出用于调试
        
        # 1. 移除addCriterion（包括各种变体）
        cleaned = re.sub(r'\s*addCriterion\s*', ' ', output)
        cleaned = re.sub(r'\s*addcriterion\s*', ' ', cleaned, flags=re.IGNORECASE)
        
        # 2. 移除action格式包装（如果存在）
        action_match = re.search(r'answer_question\(answer=["\'](.+?)["\']\)', cleaned, re.DOTALL)
        if action_match:
            cleaned = action_match.group(1)
            if self.debug:
                print(f"[ChartQAAgent] Extracted from action format")
        
        # 3. 移除可能的特殊标记
        cleaned = re.sub(r'<\|[^>]+\|>', '', cleaned)
        
        # 4. 修复重复的句子或短语
        lines = cleaned.split('\n')
        unique_lines = []
        prev_line = None
        repeat_count = 0
        
        for line in lines:
            line = line.strip()
            if line == prev_line and line:
                repeat_count += 1
                if repeat_count < 2:
                    unique_lines.append(line)
            else:
                repeat_count = 0
                if line:  # 只添加非空行
                    unique_lines.append(line)
                prev_line = line
        
        cleaned = '\n'.join(unique_lines)
        
        # 5. 清理过多的空白
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r' {2,}', ' ', cleaned)
        
        # 6. 移除孤立的中文字符
        cleaned = re.sub(r'(?<![a-zA-Z])[\u4e00-\u9fff]+(?![a-zA-Z])', '', cleaned)
        
        # 7. 确保结尾干净
        cleaned = cleaned.strip()
        
        if self.debug and original_output != cleaned:
            print(f"[ChartQAAgent] ===== OUTPUT CLEANING =====")
            print(f"[ChartQAAgent] Original length: {len(original_output)}")
            print(f"[ChartQAAgent] Cleaned length: {len(cleaned)}")
            if "addCriterion" in original_output:
                count = original_output.count("addCriterion")
                print(f"[ChartQAAgent] Removed {count} occurrences of 'addCriterion'")
            print(f"[ChartQAAgent] First 200 chars of cleaned: {cleaned[:200]}...")
        
        return cleaned
    
    def _is_yes_no_question(self, question: str) -> bool:
        """判断是否是Yes/No问题"""
        question_lower = question.lower()
        
        # 排除明显不是Yes/No的问题
        if any(phrase in question_lower for phrase in [
            'how many', 'what is', 'which', 'when', 'where', 'who', 
            'what year', 'in which year', 'what value', 'sum', 'total'
        ]):
            return False
        
        # 只有明确的Yes/No模式才返回True
        yes_no_patterns = [
            r'^is\s+',      # Is the value...
            r'^are\s+',     # Are there...
            r'^does\s+',    # Does the chart...
            r'^do\s+',      # Do the values...
            r'^was\s+',     # Was the value...
            r'^were\s+',    # Were the values...
            r'^has\s+',     # Has the value...
            r'^have\s+',    # Have the values...
            r'^can\s+',     # Can we see...
            r'^will\s+',    # Will the value...
            r'^would\s+',   # Would the value...
        ]
        
        for pattern in yes_no_patterns:
            if re.match(pattern, question_lower) and '?' in question:
                return True
        
        return False
    
    def _enhance_question_with_guidance(self, question: str) -> str:
        """根据问题类型添加分析指导"""
        question_lower = question.lower()
        
        # 1. 计数问题
        if any(phrase in question_lower for phrase in ['how many', 'count', 'number of']):
            return f"""Look at the chart and count the items that match the criteria.
Question: {question}
Answer with just a number: """
        
        # 2. 求和问题
        elif any(phrase in question_lower for phrase in ['sum', 'total', 'add']):
            return f"""Calculate the sum of the relevant values from the chart.
Question: {question}
Answer with the total: """
        
        # 3. 年份/时间问题
        elif any(phrase in question_lower for phrase in ['which year', 'what year', 'when', 'in which year']):
            return f"""Find the year or time period requested in the chart.
Question: {question}
Answer: """
        
        # 4. Yes/No问题 - 使用更严格的判断
        elif self._is_yes_no_question(question):
            return f"""Look at the chart and answer yes or no.
Question: {question}
Answer (Yes/No): """
        
        # 5. 值/数据查找问题
        elif any(phrase in question_lower for phrase in ['what is the', 'what was the', 'what value', 'which value']):
            return f"""Find the specific value in the chart.
Question: {question}
Answer: """
        
        # 6. 其他问题 - 通用提示
        else:
            return f"""Analyze the chart and answer the question.
Question: {question}
Short answer: """
    
    def act(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Enhanced act method with direct generation"""
        
        # 增强问题提示
        original_question = observation.get('question', '')
        enhanced_question = self._enhance_question_with_guidance(original_question)
        
        if self.debug:
            print(f"\n[ChartQAAgent] ===== STARTING ACT =====")
            print(f"[ChartQAAgent] Original question: {original_question}")
            print(f"[ChartQAAgent] Enhanced prompt preview: {enhanced_question[:200]}...")
        
        # ===== 直接生成 =====
        try:
            # 1. 加载图像
            image_path = observation['image_path']
            image = Image.open(image_path).convert('RGB')
            
            # 2. 准备消息
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": enhanced_question}
                ]
            }]
            
            # 3. 应用chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 4. 处理输入
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt"
            ).to(self.model.device)
            
            # 5. 生成 - 更严格的参数控制
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=64,       # 强制简洁回答
                    min_new_tokens=1,        # 允许非常短的答案
                    do_sample=False,         # 确保贪婪解码
                    temperature=0.1,         # 低温度
                    repetition_penalty=1.0,  # 暂时关闭重复惩罚
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            
            # 6. 解码
            generated_ids = outputs[:, inputs.input_ids.shape[1]:]
            raw_response = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            # 7. 清理输出
            base_response = self._clean_model_output(raw_response)
            
            if self.debug:
                print(f"\n[ChartQAAgent] ===== RAW RESPONSE =====")
                print(f"[ChartQAAgent] Length: {len(raw_response)}")
                print(f"[ChartQAAgent] First 300 chars: {raw_response[:300]}...")
                if "addCriterion" in raw_response:
                    print(f"[ChartQAAgent] WARNING: Found addCriterion in raw response!")
                print(f"\n[ChartQAAgent] ===== CLEANED RESPONSE =====")
                print(base_response[:500])
                print(f"[ChartQAAgent] ===== END RESPONSE =====")
            
            # 记录token使用情况
            tokens_generated = generated_ids.shape[1]
            tokens_used = inputs.input_ids.shape[1] + tokens_generated
            
        except Exception as e:
            print(f"[ChartQAAgent] ERROR in direct generation: {e}")
            import traceback
            traceback.print_exc()
            
            # 如果直接生成失败，尝试调用父类但清理输出
            print(f"[ChartQAAgent] Falling back to parent act method")
            action, extra_info = super().act(observation)
            
            # 清理父类返回的action
            if isinstance(action, str):
                action = self._clean_model_output(action)
            
            return action, extra_info
        
        # ===== 提取答案 =====
        answer = self._extract_final_answer(base_response)
        
        if self.debug:
            print(f"[ChartQAAgent] Extracted answer: '{answer}'")
        
        # ===== 构建返回信息 =====
        extra_info = {
            'tokens_generated': tokens_generated,
            'tokens_used': tokens_used,
            'has_actions': False,
            'model': getattr(self.config, 'model_name', 'unknown'),
            'temperature': 0.1,
            'attempt': observation.get('attempt', 1),
            'step_count': observation.get('step_count', 0),
            'base_response': base_response,
            'structured': False,
            'raw_response_length': len(raw_response),
            'cleaned': raw_response != base_response
        }
        
        # ===== 结构化推理（如果需要）=====
        if self.enable_structured_reasoning and original_question:
            question_type = self.chartqa_reasoner.classify_question(original_question)
            
            if self.debug:
                print(f"[ChartQAAgent] Question type: {question_type}")
                print(f"[ChartQAAgent] Structured reasoning enabled: {self.enable_structured_reasoning}")
            
            # 特殊处理counting问题
            if question_type == 'counting':
                import re
                # 直接从响应中提取数字
                numbers = re.findall(r'\b(\d+)\b', base_response)
                
                if self.debug:
                    print(f"[ChartQAAgent] Counting question - found numbers: {numbers}")
                
                if numbers:
                    # 对于counting问题，通常第一个数字是答案
                    # 但要排除年份（4位数）和太大的数字
                    valid_numbers = [n for n in numbers if 1 <= len(n) <= 2]  # 通常计数不会超过99
                    
                    if valid_numbers:
                        structured_answer = valid_numbers[0]
                    else:
                        # 如果没有合适的小数字，使用第一个数字
                        structured_answer = numbers[0] if numbers else answer
                    
                    if self.debug:
                        print(f"[ChartQAAgent] Using number as structured answer: '{structured_answer}'")
                        print(f"[ChartQAAgent] Marking as structured reasoning")
                    
                    extra_info.update({
                        'reasoning': f'Extracted count from response',
                        'structured': True,
                        'confidence': 0.9,  # 高置信度
                        'calculator_used': False,
                        'structured_answer': structured_answer,
                        'question_type': question_type
                    })
                    
                    return structured_answer, extra_info
                else:
                    if self.debug:
                        print(f"[ChartQAAgent] No numbers found in response for counting question")
            
            # 对于求和问题，特殊处理
            elif question_type == 'summation':
                # 从base_response中查找提到的数值
                import re
                # 查找所有百分比或数值
                pattern = r'(\d+)%?'
                matches = re.findall(pattern, base_response)
                
                if self.debug:
                    print(f"[ChartQAAgent] Summation - found values: {matches}")
                
                # 对于求和问题，通常需要找到提到的具体数值
                # 如果模型已经计算了总和，尝试找到它
                sum_patterns = [
                    r'total[^0-9]*(\d+)',
                    r'sum[^0-9]*(\d+)',
                    r'add[^0-9]*up[^0-9]*to[^0-9]*(\d+)',
                    r'=\s*(\d+)',
                    r'(\d+)\s*(?:total|sum|in total)',
                ]
                
                for pattern in sum_patterns:
                    match = re.search(pattern, base_response, re.IGNORECASE)
                    if match:
                        sum_value = match.group(1)
                        if self.debug:
                            print(f"[ChartQAAgent] Found sum value: {sum_value}")
                        
                        extra_info.update({
                            'reasoning': f'Extracted sum from response',
                            'structured': True,
                            'confidence': 0.8,
                            'calculator_used': False,
                            'structured_answer': sum_value,
                            'question_type': question_type
                        })
                        
                        return sum_value, extra_info
                
                # 如果没找到明确的和，降级到普通的推理
                try:
                    enhanced_result = self.chartqa_reasoner.reason(
                        original_question,
                        base_response,
                        question_type
                    )
                    
                    if self.debug:
                        print(f"[ChartQAAgent] Enhanced result: {enhanced_result}")
                    
                    # 对于求和，只有高置信度才使用
                    if enhanced_result.get('confidence', 0) > 0.7:
                        structured_answer = str(enhanced_result['answer'])
                        
                        # 检查答案是否合理
                        if structured_answer.isdigit() and len(structured_answer) < 6:
                            extra_info.update({
                                'reasoning': enhanced_result.get('reasoning', ''),
                                'structured': True,
                                'confidence': enhanced_result['confidence'],
                                'calculator_used': enhanced_result.get('calculator_used', False),
                                'structured_answer': structured_answer,
                                'question_type': question_type
                            })
                            
                            return structured_answer, extra_info
                            
                except Exception as e:
                    if self.debug:
                        print(f"[ChartQAAgent] Error in summation reasoning: {e}")
            
            # 对于其他支持的类型
            elif question_type in ['comparison', 'minmax', 'numerical']:
                try:
                    enhanced_result = self.chartqa_reasoner.reason(
                        original_question,
                        base_response,
                        question_type
                    )
                    
                    if self.debug:
                        print(f"[ChartQAAgent] Enhanced result: {enhanced_result}")
                        print(f"[ChartQAAgent] Confidence: {enhanced_result.get('confidence', 0)}")
                    
                    # 使用中等阈值
                    if enhanced_result.get('confidence', 0) > 0.3:
                        structured_answer = str(enhanced_result['answer'])
                        
                        # 检查答案是否合理（不应该是整个句子）
                        if len(structured_answer) < 50:  # 答案应该很短
                            if self.debug:
                                print(f"[ChartQAAgent] Using structured answer: '{structured_answer}'")
                            
                            extra_info.update({
                                'reasoning': enhanced_result.get('reasoning', ''),
                                'structured': True,
                                'confidence': enhanced_result['confidence'],
                                'calculator_used': enhanced_result.get('calculator_used', False),
                                'structured_answer': structured_answer,
                                'question_type': question_type
                            })
                            
                            return structured_answer, extra_info
                        else:
                            if self.debug:
                                print(f"[ChartQAAgent] Answer too long ({len(structured_answer)} chars), ignoring structured reasoning")
                    else:
                        if self.debug:
                            print(f"[ChartQAAgent] Confidence too low ({enhanced_result.get('confidence', 0)}), using extracted answer")
                            
                except Exception as e:
                    if self.debug:
                        print(f"[ChartQAAgent] Error in structured reasoning: {e}")
                        import traceback
                        traceback.print_exc()
            else:
                if self.debug:
                    print(f"[ChartQAAgent] Question type {question_type} not supported for structured reasoning")
        
        # 返回提取的答案
        final_answer = answer if answer else base_response
        
        if self.debug:
            print(f"[ChartQAAgent] Final answer: '{final_answer}'")
            print(f"[ChartQAAgent] ===== END ACT =====\n")
        
        return final_answer, extra_info
    
    def _extract_final_answer(self, response: str) -> str:
        """从响应中提取最终答案"""
        import re
        
        response = response.strip()
        
        # 1. 优先处理Yes/No答案
        first_word = response.split()[0].lower() if response.split() else ""
        if first_word in ['yes', 'no']:
            return first_word.capitalize()
        
        # 检查整个响应是否明确包含yes或no
        response_lower = response.lower()
        if response_lower.startswith('yes,') or response_lower.startswith('yes ') or response_lower == 'yes':
            return 'Yes'
        elif response_lower.startswith('no,') or response_lower.startswith('no ') or response_lower == 'no':
            return 'No'
        
        # 2. 先尝试提取直接的答案模式
        answer_patterns = [
            r'Answer with just a number:\s*(\d+)',  # 匹配我们的提示格式
            r'Answer with the total:\s*(\d+)',      # 求和格式
            r'Answer \(Yes/No\):\s*(\w+)',         # Yes/No格式
            r'Answer:\s*([^\n,.]+)',                # 通用答案格式
            r'Direct Answer:\s*([^\n]+)',
            r'Final answer:\s*([^\n]+)',
            r'The answer is:\s*([^\n]+)',
            r'The answer is\s+([^\n.]+)',
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip('.,!? ')
        
        # 3. 对于年份问题，查找4位数字
        if re.search(r'year|when', response, re.IGNORECASE):
            years = re.findall(r'\b(19\d{2}|20\d{2})\b', response)
            if years:
                return years[0]
        
        # 4. 对于其他数字答案，查找独立的数字
        # 优先查找句末的数字
        sentence_end_number = re.search(r'(\d+)[.,!?]?\s*$', response)
        if sentence_end_number:
            return sentence_end_number.group(1)
        
        # 查找所有数字
        numbers = re.findall(r'\b(\d+)\b', response)
        if numbers:
            # 如果有年份格式的数字，优先返回
            for num in numbers:
                if len(num) == 4 and (num.startswith('19') or num.startswith('20')):
                    return num
            # 否则返回第一个数字
            return numbers[0]
        
        # 5. 返回整个清理后的响应的第一行或前50个字符
        first_line = response.split('\n')[0] if response else response
        if len(first_line) > 50:
            return first_line[:50].strip()
        return first_line