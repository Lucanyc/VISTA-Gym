import re
import torch
from typing import Dict, Any, Tuple, Optional
from PIL import Image
from vlm_gym.agents.vlm_agent import VLMAgent
from .reasoning import ChartQAReasoner


class ChartQAAgent(VLMAgent):
    """VLM Agent enhanced with ChartQA reasoning and optional tool support via composition"""
    
    def __init__(self, config: Dict[str, Any]):
        # Extract ChartQA-specific parameters before passing to parent
        chartqa_params = {}
        chartqa_keys = ['enable_structured_reasoning', 'use_calculator', 'debug', '_chartqa_params']
        
        # Extract tool-related parameters
        tool_params = {}
        tool_keys = ['enable_tools', 'max_tool_calls', 'tool_selection_strategy', 
                     'tool_response_mode', 'deepeyes_prompt_style']
        
        # If config has _chartqa_params, use it directly
        if '_chartqa_params' in config:
            chartqa_params = config.pop('_chartqa_params')
        else:
            # Otherwise extract ChartQA specific params
            for key in chartqa_keys:
                if key in config:
                    chartqa_params[key] = config.pop(key)
        
        # Extract tool parameters
        for key in tool_keys:
            if key in config:
                tool_params[key] = config.pop(key)
        
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
        
        # Pass debug to reasoner
        self.chartqa_reasoner.debug = self.debug
        
        # ===== Tool Agent Composition =====
        self.tool_agent = None
        self.enable_tools = tool_params.get('enable_tools', False)
        
        if self.enable_tools:
            try:
                from vlm_gym.agents.vlm_agent_with_tools import VLMAgentWithTools
                
                # Prepare config for tool agent - merge base config with tool params
                tool_agent_config = {
                    **config,  # Base config (model info etc.)
                    **tool_params  # Tool specific params
                }
                
                # Create tool agent instance
                self.tool_agent = VLMAgentWithTools(tool_agent_config)
                
                # Share the same model instance to save memory
                if hasattr(self, 'model') and self.model is not None:
                    self.tool_agent.model = self.model
                if hasattr(self, 'processor') and self.processor is not None:
                    self.tool_agent.processor = self.processor
                    
                print(f"[ChartQAAgent] Tool agent initialized successfully")
                
            except Exception as e:
                print(f"[ChartQAAgent] Failed to initialize tool agent: {e}")
                self.tool_agent = None
                self.enable_tools = False
        
        # Tool usage tracking
        self.tool_usage_history = []
        self.current_tool_calls = 0
        
        # Calculation tracking
        self._current_is_calculation = False
        self.calculation_history = []
        
        # Ensure model is loaded
        self.load_model()
        
        print(f"[ChartQAAgent] Initialized with:")
        print(f"  - debug={self.debug}")
        print(f"  - structured_reasoning={self.enable_structured_reasoning}")
        print(f"  - calculator={self.use_calculator}")
        print(f"  - tools_enabled={self.enable_tools}")
        if self.enable_tools and self.tool_agent:
            print(f"  - tool_strategy={tool_params.get('tool_selection_strategy', 'adaptive')}")
    
    def load_model(self):
        """Force load model"""
        if not hasattr(self, '_loaded') or not self._loaded:
            # Call parent class load_model method
            super().load_model()
            
            # If tool_agent exists, ensure it uses the same model
            if self.tool_agent and hasattr(self, 'model'):
                self.tool_agent.model = self.model
                self.tool_agent.processor = self.processor
                self.tool_agent._loaded = True
    
    def act(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Enhanced act method with optional tool support and calculation preprocessing"""
        
        # ===== Check if this is a tool feedback response =====
        if observation.get("requires_response") and "tool_feedback" in observation:
            if self.debug:
                print(f"[ChartQAAgent] Processing tool feedback")
            
            # Use tool agent to handle feedback if available
            if self.tool_agent:
                action, extra_info = self.tool_agent.act(observation)
                
                # After getting response from zoomed image, apply ChartQA enhancements
                if '<answer>' in action:
                    # Extract answer from DeepEyes format
                    import re
                    answer_match = re.search(r'<answer>(.*?)</answer>', action, re.DOTALL)
                    if answer_match:
                        answer_text = answer_match.group(1).strip()
                        
                        # Apply ChartQA reasoning to refine the answer
                        if self.enable_structured_reasoning:
                            original_question = observation.get('question', '')
                            refined_answer = self._refine_answer_with_reasoning(
                                answer_text, 
                                original_question
                            )
                            if refined_answer != answer_text:
                                # Update the action with refined answer
                                action = action.replace(
                                    f"<answer>{answer_text}</answer>",
                                    f"<answer>{refined_answer}</answer>"
                                )
                                extra_info['chartqa_refined'] = True
                
                return action, extra_info
            else:
                # Fallback to base behavior
                return self._generate_direct_answer_from_feedback(observation)
        
        # ===== Pre-process for calculation questions =====
        question = observation.get('question', '')
        question_type = self.chartqa_reasoner.classify_question(question)
        
        # Define calculation question types
        CALCULATION_TYPES = ['summation', 'average', 'percentage', 'difference', 'ratio']
        
        # Set flag if this is a calculation question
        if question_type in CALCULATION_TYPES:
            self._current_is_calculation = True
            if self.debug:
                print(f"[ChartQAAgent] Detected calculation question: {question_type}")
                print(f"[ChartQAAgent] Calculator enabled: {self.use_calculator}")
        else:
            self._current_is_calculation = False
        
        # ===== Check if we should use tools =====
        if self.tool_agent and self.enable_tools:
            should_use, reason = self._should_use_tool(observation)
            
            if should_use:
                if self.debug:
                    print(f"[ChartQAAgent] Decided to use tool: {reason}")
                
                # Let tool agent handle the action generation
                action, extra_info = self.tool_agent.act(observation)
                
                # Track tool usage
                self.current_tool_calls += 1
                self.tool_usage_history.append({
                    'step': observation.get('step', 0),
                    'reason': reason,
                    'action': action
                })
                
                # Add ChartQA specific info
                extra_info['chartqa_tool_decision'] = reason
                extra_info['chartqa_tool_calls'] = self.current_tool_calls
                
                return action, extra_info
        
        # ===== Otherwise, use ChartQA's direct generation =====
        return self._chartqa_direct_generation(observation)
    
    def _should_use_tool(self, observation: Dict[str, Any]) -> Tuple[bool, str]:
        """Decide whether to use DeepEyes tool based on question and context"""
        
        # Don't use tools if already at max calls
        max_tool_calls = getattr(self.tool_agent, 'max_tool_calls', 5) if self.tool_agent else 5
        if self.current_tool_calls >= max_tool_calls:
            return False, "max_tool_calls_reached"
        
        question = observation.get('question', '').lower()
        attempt = observation.get('attempt', 1)
        
        # Strategy 1: Use tools for questions that mention difficulty seeing
        if any(phrase in question for phrase in [
            'hard to see', 'difficult to read', 'small', 'tiny', 
            'unclear', 'blurry', 'precise value', 'exact number'
        ]):
            return True, "explicit_visibility_issue"
        
        # Strategy 2: For counting questions with many items
        if ('count' in question or 'how many' in question):
            # On retry, consider using tools
            if attempt > 1:
                return True, "counting_retry_attempt"
            # For complex counting (e.g., "below 40")
            if any(word in question for word in ['below', 'above', 'between', 'less than', 'greater than']):
                return True, "complex_counting_criteria"
        
        # Strategy 3: For questions requiring precise value reading
        if any(phrase in question for phrase in [
            'exact value', 'specific number', 'precise', 
            'what is the value', 'what was the value'
        ]):
            # Especially on retry
            if attempt > 1:
                return True, "precise_value_retry"
        
        # Strategy 4: For summation/calculation questions that might need to see all values clearly
        if any(phrase in question for phrase in ['sum', 'total', 'add up', 'aggregate']):
            if attempt > 1:  # If first attempt failed
                return True, "summation_retry"
        
        # Strategy 5: For comparison questions that need to see multiple values
        if any(phrase in question for phrase in ['compare', 'difference between', 'which is higher', 'which is lower']):
            if 'multiple' in question or 'all' in question or attempt > 1:
                return True, "comparison_multiple_values"
        
        # Default: Don't use tools on first attempt for most questions
        return False, "no_tool_needed"
    
    def _generate_direct_answer_from_feedback(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Generate answer based on tool feedback when tool agent is not available"""
        # This is a fallback method
        tool_feedback = observation.get('tool_feedback', {})
        original_question = observation.get('question', '')
        
        # Simple prompt for answering based on zoomed view
        enhanced_prompt = f"Based on the zoomed view, answer: {original_question}\nAnswer:"
        
        # Create a modified observation for parent's act method
        modified_obs = observation.copy()
        modified_obs['question'] = enhanced_prompt
        
        # Use parent's act method
        action, extra_info = super().act(modified_obs)
        
        # Format as DeepEyes answer
        formatted_action = f"<answer>{action.strip()}</answer>"
        
        extra_info['fallback_method'] = True
        
        return formatted_action, extra_info
    
    def _refine_answer_with_reasoning(self, answer: str, question: str) -> str:
        """Apply ChartQA reasoning to refine an answer"""
        if not self.enable_structured_reasoning:
            return answer
        
        try:
            # Classify question
            question_type = self.chartqa_reasoner.classify_question(question)
            
            # For certain question types, apply refinement
            if question_type in ['counting', 'summation', 'average', 'percentage']:
                # Create a simple context for reasoning
                enhanced_result = self.chartqa_reasoner.reason(
                    question,
                    f"The answer appears to be {answer}",
                    question_type
                )
                
                if enhanced_result.get('confidence', 0) > 0.7:
                    refined = str(enhanced_result['answer'])
                    if self._validate_answer(refined, question_type):
                        return refined
            
        except Exception as e:
            if self.debug:
                print(f"[ChartQAAgent] Error in answer refinement: {e}")
        
        return answer
    
    def _preprocess_for_calculation(self, question: str, response: str) -> str:
        """Preprocess response to be more suitable for calculator processing"""
        
        question_type = self.chartqa_reasoner.classify_question(question)
        
        if self.debug:
            print(f"[ChartQAAgent] Preprocessing for {question_type} calculation")
        
        # For summation questions, if response doesn't have explicit value list, try to extract
        if question_type == 'summation' and 'sum' in question.lower():
            # Check if specific year values are needed
            year_match = re.search(r'in (?:the years? )?(\d{4})(?: and (\d{4}))?', question)
            if year_match:
                years = [year_match.group(1)]
                if year_match.group(2):
                    years.append(year_match.group(2))
                
                # Find values for these years in response
                enhanced_response = response
                for year in years:
                    # Look for "2014: 51" or "51 in 2014" format
                    patterns = [
                        rf'{year}[:\s]+(\d+)',
                        rf'(\d+)\s+in\s+{year}',
                        rf'value\s+(?:was|is)\s+(\d+)\s+in\s+{year}',
                        rf'favorable\s+(?:value\s+)?in\s+{year}\s*:?\s*(\d+)',
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, response, re.IGNORECASE)
                        if match:
                            value = match.group(1)
                            enhanced_response += f"\nValue for {year}: {value}"
                            if self.debug:
                                print(f"[ChartQAAgent] Found value {value} for year {year}")
                
                return enhanced_response
        
        # For average questions, ensure all relevant values are clearly labeled
        elif question_type == 'average':
            # Look for conditions (e.g., "above 50")
            condition_match = re.search(r'(?:above|below|over|under)\s+(\d+)', question, re.IGNORECASE)
            if condition_match:
                threshold = condition_match.group(1)
                enhanced_response = response + f"\nLooking for values {'above' if 'above' in question.lower() or 'over' in question.lower() else 'below'} {threshold}"
        
        return response
    
    def _validate_calculation_result(self, result: Dict[str, Any], question: str) -> bool:
        """Validate whether calculation result is reasonable"""
        
        if not result.get('calculator_used', False):
            return True  # Non-calculation result, skip validation
        
        answer = result.get('answer', '')
        
        try:
            # Try to extract numeric value
            num_value = float(re.sub(r'[^\d.-]', '', str(answer)))
            
            # Basic sanity check
            question_lower = question.lower()
            
            if self.debug:
                print(f"[ChartQAAgent] Validating calculation result: {answer} (numeric: {num_value})")
            
            # Percentages should be between 0-100 (in most cases)
            if 'percent' in question_lower and '%' in str(answer):
                if num_value < 0 or num_value > 200:  # Allow some cases above 100%
                    if self.debug:
                        print(f"[ChartQAAgent] Invalid percentage: {num_value}")
                    return False
            
            # Counting questions usually don't yield very large numbers
            if 'how many' in question_lower and num_value > 1000:
                if self.debug:
                    print(f"[ChartQAAgent] Unrealistic count: {num_value}")
                return False
            
            # Years should be in a reasonable range
            if 'year' in question_lower and (num_value < 1900 or num_value > 2100):
                if self.debug:
                    print(f"[ChartQAAgent] Invalid year: {num_value}")
                return False
            
            # Sum results should not be negative
            if 'sum' in question_lower and num_value < 0:
                if self.debug:
                    print(f"[ChartQAAgent] Negative sum: {num_value}")
                return False
            
        except:
            # If cannot convert to number, may be a non-numerical answer
            pass
        
        return True
    
    def _chartqa_direct_generation(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """ChartQA's original direct generation logic with calculation enhancements"""
        
        # Enhance question prompt
        original_question = observation.get('question', '')
        enhanced_question = self._enhance_question_with_guidance(original_question)
        
        if self.debug:
            print(f"\n[ChartQAAgent] ===== STARTING DIRECT GENERATION =====")
            print(f"[ChartQAAgent] Original question: {original_question}")
            print(f"[ChartQAAgent] Enhanced prompt preview: {enhanced_question[:200]}...")
            if self._current_is_calculation:
                print(f"[ChartQAAgent] This is a CALCULATION question")
        
        # ===== Direct generation =====
        try:
            # 1. Load image
            image_path = observation['image_path']
            image = Image.open(image_path).convert('RGB')
            
            # 2. Prepare messages
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": enhanced_question}
                ]
            }]
            
            # 3. Apply chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 4. Process inputs
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt"
            ).to(self.model.device)
            
            # 5. Generate - stricter parameter control
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=64,       # Force concise answer
                    min_new_tokens=1,        # Allow very short answers
                    do_sample=False,         # Ensure greedy decoding
                    temperature=0.1,         # Low temperature
                    repetition_penalty=1.0,  # Disable repetition penalty for now
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            
            # 6. Decode
            generated_ids = outputs[:, inputs.input_ids.shape[1]:]
            raw_response = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            # 7. Clean output
            base_response = self._clean_model_output(raw_response)
            
            # 8. If calculation question, preprocess response
            if self._current_is_calculation:
                base_response = self._preprocess_for_calculation(original_question, base_response)
            
            if self.debug:
                print(f"\n[ChartQAAgent] ===== RAW RESPONSE =====")
                print(f"[ChartQAAgent] Length: {len(raw_response)}")
                print(f"[ChartQAAgent] First 300 chars: {raw_response[:300]}...")
                print(f"\n[ChartQAAgent] ===== CLEANED RESPONSE =====")
                print(base_response[:500])
            
            # Record token usage
            tokens_generated = generated_ids.shape[1]
            tokens_used = inputs.input_ids.shape[1] + tokens_generated
            
        except Exception as e:
            print(f"[ChartQAAgent] ERROR in direct generation: {e}")
            import traceback
            traceback.print_exc()
            
            # If direct generation fails, call parent class
            print(f"[ChartQAAgent] Falling back to parent act method")
            action, extra_info = super().act(observation)
            
            # Clean action returned by parent class
            if isinstance(action, str):
                action = self._clean_model_output(action)
            
            extra_info['chartqa_tool_available'] = self.enable_tools
            extra_info['chartqa_tool_calls'] = self.current_tool_calls
            
            return action, extra_info
        
        # ===== Extract answer =====
        answer = self._extract_final_answer(base_response)
        
        if self.debug:
            print(f"[ChartQAAgent] Extracted answer: '{answer}'")
        
        # ===== Build return info =====
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
            'cleaned': raw_response != base_response,
            'chartqa_tool_available': self.enable_tools,
            'chartqa_tool_calls': self.current_tool_calls,
            'is_calculation_question': self._current_is_calculation
        }
        
        # ===== Structured reasoning (if needed) =====
        if self.enable_structured_reasoning and original_question:
            question_type = self.chartqa_reasoner.classify_question(original_question)
            
            if self.debug:
                print(f"[ChartQAAgent] Question type: {question_type}")
                print(f"[ChartQAAgent] Applying structured reasoning...")
            
            # Apply structured reasoning based on question type
            structured_result = self._apply_structured_reasoning(
                question_type, original_question, base_response, answer
            )
            
            if structured_result:
                answer = structured_result['answer']
                extra_info.update(structured_result['extra_info'])
                
                # Record calculation history
                if structured_result['extra_info'].get('calculator_used', False):
                    self.calculation_history.append({
                        'question': original_question,
                        'question_type': question_type,
                        'calculation': structured_result['extra_info'].get('reasoning', ''),
                        'result': answer
                    })
        
        # Return final answer
        final_answer = answer if answer else base_response
        
        if self.debug:
            print(f"[ChartQAAgent] Final answer: '{final_answer}'")
            print(f"[ChartQAAgent] ===== END DIRECT GENERATION =====\n")
        
        return final_answer, extra_info
    
    def _apply_structured_reasoning(self, question_type: str, question: str, 
                                   response: str, extracted_answer: str) -> Optional[Dict[str, Any]]:
        """Apply structured reasoning based on question type with enhanced calculation support"""
        
        if self.debug:
            print(f"[ChartQAAgent] _apply_structured_reasoning called")
            print(f"  - Question type: {question_type}")
            print(f"  - Calculator enabled: {self.use_calculator}")
            print(f"  - Extracted answer: '{extracted_answer}'")
        
        # Define all supported types
        SUPPORTED_TYPES = [
            'counting', 'summation', 'average', 'percentage', 
            'difference', 'ratio', 'comparison', 'minmax', 
            'numerical', 'retrieval', 'trend', 'other'
        ]
        
        # Define calculation question types
        CALCULATION_TYPES = ['summation', 'average', 'percentage', 'difference', 'ratio']
        
        if question_type not in SUPPORTED_TYPES:
            return None
        
        # For calculation questions, force use of reasoner
        if question_type in CALCULATION_TYPES and self.use_calculator:
            if self.debug:
                print(f"[ChartQAAgent] Forcing calculation for {question_type} question")
            
            # Even if response already has an answer, verify calculation through reasoner
            enhanced_result = self.chartqa_reasoner.reason(
                question,
                response,
                question_type
            )
            
            if self.debug:
                print(f"[ChartQAAgent] Reasoner result: {enhanced_result}")
            
            # If calculator was used, validate the result
            if enhanced_result.get('calculator_used', False):
                # Validate calculation result
                if self._validate_calculation_result(enhanced_result, question):
                    # If validation passes, trust the calculation result
                    return {
                        'answer': str(enhanced_result['answer']),
                        'extra_info': {
                            'reasoning': enhanced_result.get('reasoning', ''),
                            'structured': True,
                            'confidence': enhanced_result.get('confidence', 0.9),
                            'calculator_used': True,
                            'structured_answer': enhanced_result['answer'],
                            'question_type': question_type,
                            'calculation_details': enhanced_result.get('reasoning', '')
                        }
                    }
                else:
                    if self.debug:
                        print(f"[ChartQAAgent] Calculation result failed validation")
        
        try:
            # Special handling for counting
            if question_type == 'counting':
                import re
                numbers = re.findall(r'\b(\d+)\b', response)
                
                if self.debug:
                    print(f"[ChartQAAgent] Counting question - found numbers: {numbers}")
                
                if numbers:
                    # Filter out years and very large numbers
                    valid_numbers = [n for n in numbers if 1 <= len(n) <= 2]
                    
                    if valid_numbers:
                        structured_answer = valid_numbers[0]
                    else:
                        structured_answer = numbers[0] if numbers else extracted_answer
                    
                    return {
                        'answer': structured_answer,
                        'extra_info': {
                            'reasoning': 'Extracted count from response',
                            'structured': True,
                            'confidence': 0.9,
                            'calculator_used': False,
                            'structured_answer': structured_answer,
                            'question_type': question_type
                        }
                    }
            
            # For other types, use reasoner
            enhanced_result = self.chartqa_reasoner.reason(
                question,
                response,
                question_type
            )
            
            if self.debug:
                print(f"[ChartQAAgent] Enhanced result: {enhanced_result}")
            
            # Set confidence thresholds - lower thresholds for calculation questions
            confidence_thresholds = {
                'summation': 0.4,      # Lower threshold
                'average': 0.3,        # Lower threshold
                'percentage': 0.3,     # Lower threshold
                'difference': 0.3,     # Lower threshold
                'ratio': 0.3,          # Lower threshold
                'comparison': 0.5,
                'minmax': 0.5,
                'numerical': 0.4,
                'retrieval': 0.5,
                'trend': 0.5,
                'other': 0.3
            }
            
            threshold = confidence_thresholds.get(question_type, 0.5)
            
            if enhanced_result.get('confidence', 0) > threshold:
                structured_answer = str(enhanced_result['answer'])
                
                # Validate answer
                if self._validate_answer(structured_answer, question_type):
                    return {
                        'answer': structured_answer,
                        'extra_info': {
                            'reasoning': enhanced_result.get('reasoning', ''),
                            'structured': True,
                            'confidence': enhanced_result['confidence'],
                            'calculator_used': enhanced_result.get('calculator_used', False),
                            'structured_answer': structured_answer,
                            'question_type': question_type
                        }
                    }
                    
        except Exception as e:
            if self.debug:
                print(f"[ChartQAAgent] Error in structured reasoning: {e}")
                import traceback
                traceback.print_exc()
        
        return None
    
    def reset(self):
        """Reset agent state"""
        super().reset()
        
        # Reset tool usage tracking
        self.tool_usage_history = []
        self.current_tool_calls = 0
        
        # Reset calculation tracking
        self._current_is_calculation = False
        self.calculation_history = []
        
        # Reset tool agent if exists
        if self.tool_agent:
            self.tool_agent.reset()
    
    # ===== Keep all existing helper methods unchanged =====
    
    def _clean_model_output(self, output: str) -> str:
        """Clean abnormal characters and patterns from model output"""
        if not output:
            return output
            
        original_output = output  # Save original output for debugging
        
        # 1. Remove addCriterion (including various variants)
        cleaned = re.sub(r'\s*addCriterion\s*', ' ', output)
        cleaned = re.sub(r'\s*addcriterion\s*', ' ', cleaned, flags=re.IGNORECASE)
        
        # 2. Remove action format wrapper (if present)
        action_match = re.search(r'answer_question\(answer=["\'](.+?)["\']\)', cleaned, re.DOTALL)
        if action_match:
            cleaned = action_match.group(1)
            if self.debug:
                print(f"[ChartQAAgent] Extracted from action format")
        
        # 3. Remove possible special tokens
        cleaned = re.sub(r'<\|[^>]+\|>', '', cleaned)
        
        # 4. Fix repeated sentences or phrases
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
                if line:  # Only add non-empty lines
                    unique_lines.append(line)
                prev_line = line
        
        cleaned = '\n'.join(unique_lines)
        
        # 5. Clean excessive whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r' {2,}', ' ', cleaned)
        
        # 6. Remove isolated Chinese characters
        cleaned = re.sub(r'(?<![a-zA-Z])[\u4e00-\u9fff]+(?![a-zA-Z])', '', cleaned)
        
        # 7. Ensure clean ending
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
        """Determine if question is a Yes/No question"""
        question_lower = question.lower()
        
        # Exclude questions that are clearly not Yes/No
        if any(phrase in question_lower for phrase in [
            'how many', 'what is', 'which', 'when', 'where', 'who', 
            'what year', 'in which year', 'what value', 'sum', 'total'
        ]):
            return False
        
        # Only return True for explicit Yes/No patterns
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
        """Add analysis guidance based on question type"""
        question_lower = question.lower()
        
        # 1. Counting questions
        if any(phrase in question_lower for phrase in ['how many', 'count', 'number of']):
            return f"""Look at the chart and count the items that match the criteria.
Question: {question}
Answer with just a number: """
        
        # 2. Summation questions
        elif any(phrase in question_lower for phrase in ['sum', 'total', 'add']):
            return f"""Calculate the sum of the relevant values from the chart.
Question: {question}
Answer with the total: """
        
        # 3. Year/time questions
        elif any(phrase in question_lower for phrase in ['which year', 'what year', 'when', 'in which year']):
            return f"""Find the year or time period requested in the chart.
Question: {question}
Answer: """
        
        # 4. Yes/No questions - use stricter judgment
        elif self._is_yes_no_question(question):
            return f"""Look at the chart and answer yes or no.
Question: {question}
Answer (Yes/No): """
        
        # 5. Value/data lookup questions
        elif any(phrase in question_lower for phrase in ['what is the', 'what was the', 'what value', 'which value']):
            return f"""Find the specific value in the chart.
Question: {question}
Answer: """
        
        # 6. Other questions - generic prompt
        else:
            return f"""Analyze the chart and answer the question.
Question: {question}
Short answer: """
    
    def _extract_final_answer(self, response: str) -> str:
        """Extract final answer from response"""
        import re
        
        response = response.strip()
        
        # 1. Handle Yes/No answer first
        first_word = response.split()[0].lower() if response.split() else ""
        if first_word in ['yes', 'no']:
            return first_word.capitalize()
        
        # Check if the entire response explicitly contains yes or no
        response_lower = response.lower()
        if response_lower.startswith('yes,') or response_lower.startswith('yes ') or response_lower == 'yes':
            return 'Yes'
        elif response_lower.startswith('no,') or response_lower.startswith('no ') or response_lower == 'no':
            return 'No'
        
        # 2. Try to extract direct answer patterns first
        answer_patterns = [
            r'Answer with just a number:\s*(\d+)',  # Match our prompt format
            r'Answer with the total:\s*(\d+)',      # Summation format
            r'Answer \(Yes/No\):\s*(\w+)',          # Yes/No format
            r'Answer:\s*([^\n,.]+)',                 # Generic answer format
            r'Direct Answer:\s*([^\n]+)',
            r'Final answer:\s*([^\n]+)',
            r'The answer is:\s*([^\n]+)',
            r'The answer is\s+([^\n.]+)',
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip('.,!? ')
        
        # 3. For year questions, find 4-digit numbers
        if re.search(r'year|when', response, re.IGNORECASE):
            years = re.findall(r'\b(19\d{2}|20\d{2})\b', response)
            if years:
                return years[0]
        
        # 4. For other numerical answers, find standalone numbers
        # Prefer numbers at end of sentence
        sentence_end_number = re.search(r'(\d+)[.,!?]?\s*$', response)
        if sentence_end_number:
            return sentence_end_number.group(1)
        
        # Find all numbers
        numbers = re.findall(r'\b(\d+)\b', response)
        if numbers:
            # If there are year-format numbers, return those first
            for num in numbers:
                if len(num) == 4 and (num.startswith('19') or num.startswith('20')):
                    return num
            # Otherwise return the first number
            return numbers[0]
        
        # 5. Return the first line or first 50 characters of the cleaned response
        first_line = response.split('\n')[0] if response else response
        if len(first_line) > 50:
            return first_line[:50].strip()
        return first_line
    
    def _validate_answer(self, answer: str, question_type: str) -> bool:
        """Validate whether answer is reasonable"""
        if not answer:
            return False
        
        # Answer should not be too long (except for retrieval and other types)
        if question_type not in ['retrieval', 'other'] and len(answer) > 50:
            return False
        
        # Numerical types should be able to extract a number
        if question_type in ['counting', 'summation', 'average', 'percentage', 'difference', 'ratio', 'numerical']:
            import re
            # Check if it contains a number
            if not re.search(r'\d+', answer):
                return False
            
            # Avoid returning entire sentences (check if too many words)
            if ' ' in answer and len(answer.split()) > 5:
                return False
            
            # For pure numerical types, ensure it's mainly numbers
            if question_type != 'percentage':  # Percentages may include % symbol
                # After removing spaces and punctuation, should be mostly numbers
                cleaned = re.sub(r'[^\d.]', '', answer)
                if len(cleaned) < len(answer) * 0.5:  # At least 50% should be numbers
                    return False
        
        # Avoid returning meta-information containing "question" or "answer"
        if any(meta in answer.lower() for meta in ['question', 'answer is', 'the answer']):
            return False
        
        # Special handling for Yes/No questions
        if answer.lower() in ['yes', 'no']:
            return True
        
        return True
