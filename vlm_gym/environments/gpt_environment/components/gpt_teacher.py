# vlm_gym/environments/gpt_environment/components/gpt_teacher.py
#这里只有这个脚本是真的调用GPT模型

import json
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import time
from enum import Enum
import re
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from ..prompts.teaching_prompts import TeachingPrompts
from ..prompts.task_prompts import TaskPrompts
from ..exceptions import GPTAPIError, PromptGenerationError

logger = logging.getLogger(__name__)


class TeachingStyle(Enum):
    """Teaching style types"""
    SOCRATIC = "socratic"
    DIRECT = "direct"
    SCAFFOLDING = "scaffolding"
    GUIDED_DISCOVERY = "guided_discovery"
    EXPLORATORY = "exploratory"


@dataclass
class TeachingContext:
    """Context for generating teaching responses"""
    task_type: str
    task_difficulty: float
    student_level: str
    current_step: Optional[str] = None
    misconceptions: List[str] = None
    progress: float = 0.0


class GPTTeacher:
    """
    GPT-powered teacher that generates intelligent teaching prompts
    and guides students through reasoning tasks.
    
    Works with all VLMGym task types: figureqa, chartqa, clevr, geometry3k,
    geoqa, iconqa, scienceqa, mathvista, olympiadbench
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize GPT Teacher.
        
        Args:
            config: Configuration containing:
                - model: GPT model to use (e.g., 'gpt-4')
                - api_key: OpenAI API key
                - temperature: Generation temperature
                - max_tokens: Maximum tokens per response
                - system_prompt: Base system prompt
                - timeout: API timeout in seconds
        """
        self.model = config.get('model', 'gpt-4')
        self.temperature = config.get('temperature', 0.3)
        self.max_tokens = config.get('max_tokens', 500)
        self.timeout = config.get('timeout', 30)
        
        # Initialize OpenAI client
        openai.api_key = config.get('api_key')
        if not openai.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Load prompt templates
        self.teaching_prompts = TeachingPrompts()
        self.task_prompts = TaskPrompts()
        
        # System prompt
        self.system_prompt = config.get('system_prompt', self._get_default_system_prompt())
        
        # Teaching style configurations
        self.teaching_styles = self._initialize_teaching_styles()
        
        # Response cache for efficiency
        self.response_cache = {}
        self.cache_enabled = config.get('enable_cache', True)
        
        # Statistics
        self.api_calls = 0
        self.cache_hits = 0
    
    
    def _validate_messages(self, messages):
        """确保所有消息内容都是有效字符串"""
        validated_messages = []
        for msg in messages:
            content = msg.get('content', '')
            if content is None:
                content = ''
            
            validated_msg = {
                'role': msg.get('role', 'user'),
                'content': str(content)
            }
            validated_messages.append(validated_msg)
        return validated_messages
    
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for GPT teacher"""
        return """You are an helpful assistant good at solving math problems evaluating student answers. 
Your role is to:
1. Check if the student's answer is correct
2. If correct: Simply acknowledge and stop
3. If incorrect: Guide them WITHOUT revealing the answer
4. Be clear and concise
5. Focus on accuracy, not unnecessary questions
"""
    
    def _initialize_teaching_styles(self) -> Dict[TeachingStyle, Dict[str, Any]]:
        """Initialize different teaching style configurations"""
        return {
            TeachingStyle.SOCRATIC: {
                "description": "Guide through questions",
                "question_ratio": 0.8,
                "directness": 0.2,
                "example_prompt": "What do you notice about the relationship between these objects?"
            },
            TeachingStyle.DIRECT: {
                "description": "Clear, direct instruction",
                "question_ratio": 0.3,
                "directness": 0.8,
                "example_prompt": "To solve this, first identify all objects, then analyze their positions."
            },
            TeachingStyle.SCAFFOLDING: {
                "description": "Progressive support",
                "question_ratio": 0.5,
                "directness": 0.5,
                "example_prompt": "You've identified the objects correctly. Now, let's think about their spatial relationships."
            },
            TeachingStyle.GUIDED_DISCOVERY: {
                "description": "Lead to self-discovery",
                "question_ratio": 0.6,
                "directness": 0.3,
                "example_prompt": "Interesting observation! What pattern do you see emerging?"
            },
            TeachingStyle.EXPLORATORY: {
                "description": "Open-ended exploration",
                "question_ratio": 0.7,
                "directness": 0.2,
                "example_prompt": "What different ways could you approach this problem?"
            }
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def _call_gpt(self, messages: List[Dict[str, str]], 
                  temperature: Optional[float] = None,
                  max_tokens: Optional[int] = None) -> str:
        """
        Make API call to GPT with retry logic.
        
        Args:
            messages: Conversation messages
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            Generated response text
        """
        try:
            self.api_calls += 1

            logger.info(f"[DEBUG] Calling GPT API - Call #{self.api_calls}")
            logger.info(f"[DEBUG] Model: {self.model}")
            logger.info(f"[DEBUG] Number of messages: {len(messages)}")
            for i, msg in enumerate(messages):
                content = msg.get('content', '')
                if content is None:
                    content = ''
                logger.info(f"[DEBUG] Message {i}: role={msg.get('role')}, content_length={len(content)}")
            
            # 打印完整的消息内容
            for i, msg in enumerate(messages):
                content = msg.get('content') or ''
                logger.info(f"[DEBUG] Message {i}: role={msg.get('role')}, content_length={len(content)}")
                # 打印完整内容
                logger.info(f"[DEBUG] Message {i} FULL CONTENT:\n{content}\n")
            
                
            messages = self._validate_messages(messages)
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )
            
            response_text = response.choices[0].message.content.strip()
            logger.info(f"[DEBUG] GPT Response length: {len(response_text)} chars")
            logger.info(f"[DEBUG] GPT Response preview: {response_text[:100]}...")
            
            #return response.choices[0].message.content.strip()
            return response_text
        
        except Exception as e:
            error_message = str(e)
            logger.error(f"[DEBUG] GPT API failed with error: {error_message}")
            if "rate_limit" in error_message.lower():
                logger.warning(f"Rate limit hit: {e}")
                raise GPTAPIError(f"Rate limit: {e}")
            else:
                logger.error(f"OpenAI API error: {e}")
                raise GPTAPIError(f"API error: {e}")
    
    def _extract_task_info(self, task: Any) -> Dict[str, Any]:
        """
        Extract information from any task object.
        Compatible with all VLMGym task types.
        
        Args:
            task: Task object from any of the 9 task types
            
        Returns:
            Unified task information dictionary
        """
        # Extract basic information using common attribute names
        task_info = {
            'id': self._get_task_attribute(task, ['task_id', 'id', 'question_id', 'problem_id'], 'unknown'),
            'type': self._infer_task_type(task),
            'question': self._get_task_attribute(task, ['question', 'question_text', 'problem', 'prompt'], ''),
            'answer': self._get_task_attribute(task, ['answer', 'solution', 'correct_answer', 'ground_truth'], None),
            'image_path': self._get_task_attribute(task, ['image_path', 'image', 'figure_path', 'img_path'], ''),
            'metadata': {}
        }
        
        # Add task-specific metadata
        task_type = task_info['type']
        
        if task_type == 'figureqa':
            task_info['metadata'] = {
                'chart_type': getattr(task, 'chart_type', 'unknown'),
                'has_colors': hasattr(task, 'color1_name') and hasattr(task, 'color2_name'),
                'color1': getattr(task, 'color1_name', None),
                'color2': getattr(task, 'color2_name', None)
            }
        elif task_type == 'chartqa':
            task_info['metadata'] = {
                'requires_calculation': self._check_calculation_needed(task_info['question'])
            }
        elif task_type == 'clevr':
            task_info['metadata'] = {
                'question_type': self._classify_clevr_question(task_info['question'])
            }
        elif task_type in ['geometry3k', 'geoqa']:
            task_info['metadata'] = {
                'geometry_type': 'geometric_reasoning',
                'has_diagram': bool(task_info['image_path'])
            }
        
        return task_info
    
    def _get_task_attribute(self, task: Any, attribute_names: List[str], default: Any) -> Any:
        """Get attribute from task object trying multiple possible names"""
        for attr in attribute_names:
            if hasattr(task, attr):
                return getattr(task, attr)
        return default
    
    def _infer_task_type(self, task: Any) -> str:
        """Infer task type from task object"""
        class_name = task.__class__.__name__.lower()
        
        # Map class names to task types
        task_types = ['figureqa', 'chartqa', 'clevr', 'geometry3k', 'geoqa', 
                     'iconqa', 'scienceqa', 'mathvista', 'olympiadbench']
        
        for task_type in task_types:
            if task_type in class_name:
                return task_type
        
        # Try module name
        module_name = task.__class__.__module__.lower()
        for task_type in task_types:
            if task_type in module_name:
                return task_type
        
        return 'unknown'
    
    def _check_calculation_needed(self, question: str) -> bool:
        """Check if question requires calculation"""
        if not question:
            return False
        question_lower = question.lower()
        calc_keywords = ['sum', 'total', 'average', 'mean', 'difference', 'ratio', 
                        'percentage', 'calculate', 'compute', 'how many', 'how much']
        return any(keyword in question_lower for keyword in calc_keywords)
    
    def _classify_clevr_question(self, question: str) -> str:
        """Classify CLEVR question type"""
        if not question:
            return 'unknown'
        question_lower = question.lower()
        
        if 'how many' in question_lower or 'count' in question_lower:
            return 'counting'
        elif any(word in question_lower for word in ['left', 'right', 'behind', 'front']):
            return 'spatial'
        elif any(word in question_lower for word in ['color', 'shape', 'size', 'material']):
            return 'attribute'
        else:
            return 'logical'
    
    def generate_initial_prompt(self, task: Any, strategy: Any, 
                              student_level: str = "intermediate") -> str:
        """
        Generate initial teaching prompt for any task type.
        
        Args:
            task: Task object from any VLMGym task type
            strategy: Teaching strategy to use
            student_level: Student's skill level
            
        Returns:
            Initial teaching prompt
        """
        # Extract task information
        task_info = self._extract_task_info(task)
        
        # Check cache
        cache_key = f"initial_{task_info['id']}_{strategy.name}_{student_level}"
        if self.cache_enabled and cache_key in self.response_cache:
            self.cache_hits += 1
            return self.response_cache[cache_key]
        
        # Get task-specific teaching approach
        teaching_approach = self._get_teaching_approach(
            task_type=task_info['type'],
            metadata=task_info['metadata'],
            student_level=student_level
        )
        
        # Build prompt
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._build_initial_prompt_request(
                task_info=task_info,
                strategy=strategy,
                student_level=student_level,
                teaching_approach=teaching_approach
            )}
        ]
        
        try:
            response = self._call_gpt(messages)
            
            # Cache response
            if self.cache_enabled:
                self.response_cache[cache_key] = response
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate initial prompt: {e}")
            return self._get_fallback_initial_prompt(task_info, strategy)
    
    def _get_teaching_approach(self, task_type: str, metadata: Dict[str, Any], 
                             student_level: str) -> Dict[str, Any]:
        """Get task-specific teaching approach"""
        approaches = {
            'figureqa': {
                'focus': 'visual comparison and data analysis',
                'key_skills': ['chart reading', 'value comparison', 'color differentiation'],
                'common_mistakes': ['misreading axes', 'color confusion', 'incomplete scanning']
            },
            'chartqa': {
                'focus': 'data extraction and interpretation',
                'key_skills': ['chart comprehension', 'value reading', 'calculation'],
                'common_mistakes': ['unit errors', 'scale misreading', 'calculation mistakes']
            },
            'clevr': {
                'focus': 'logical reasoning about objects',
                'key_skills': ['object identification', 'attribute filtering', 'spatial reasoning'],
                'common_mistakes': ['incomplete counting', 'attribute confusion', 'spatial errors']
            },
            'geometry3k': {
                'focus': 'geometric reasoning and calculation',
                'key_skills': ['theorem application', 'diagram analysis', 'algebraic manipulation'],
                'common_mistakes': ['missing given info', 'theorem misapplication', 'calculation errors']
            },
            'geoqa': {
                'focus': 'geometric problem solving',
                'key_skills': ['angle relationships', 'area calculation', 'shape properties'],
                'common_mistakes': ['overlooking constraints', 'formula errors']
            },
            'iconqa': {
                'focus': 'visual reasoning for young learners',
                'key_skills': ['pattern recognition', 'counting', 'basic comparisons'],
                'common_mistakes': ['distraction by details', 'counting errors']
            },
            'scienceqa': {
                'focus': 'scientific reasoning',
                'key_skills': ['hypothesis testing', 'data interpretation', 'concept application'],
                'common_mistakes': ['ignoring context', 'overgeneralization']
            },
            'mathvista': {
                'focus': 'mathematical visual reasoning',
                'key_skills': ['equation solving', 'graph interpretation', 'pattern analysis'],
                'common_mistakes': ['notation errors', 'graph misreading']
            },
            'olympiadbench': {
                'focus': 'advanced problem solving',
                'key_skills': ['complex reasoning', 'proof techniques', 'creative approaches'],
                'common_mistakes': ['missing edge cases', 'incomplete arguments']
            }
        }
        
        return approaches.get(task_type, {
            'focus': 'general visual reasoning',
            'key_skills': ['observation', 'analysis', 'logical thinking'],
            'common_mistakes': ['hasty conclusions', 'missing details']
        })
    
    def _build_initial_prompt_request(self, task_info: Dict[str, Any], strategy: Any,
                                    student_level: str, teaching_approach: Dict[str, Any]) -> str:
        """Build prompt request for GPT"""
        return f"""Generate an initial teaching prompt for this visual reasoning task:

TASK INFORMATION:
- Type: {task_info['type']}
- Question: {task_info['question']}
- Student Level: {student_level}
- Teaching Strategy: {strategy.name}

TEACHING APPROACH:
- Focus: {teaching_approach.get('focus', 'visual reasoning')}
- Key Skills: {', '.join(teaching_approach.get('key_skills', []))}
- Common Mistakes: {', '.join(teaching_approach.get('common_mistakes', [])[:2])}

REQUIREMENTS:
1. Match the {strategy.name} teaching style
2. Be appropriate for {student_level} level
3. Be engaging and encouraging
4. Focus on developing reasoning skills
5. Keep initial prompt concise (2-3 sentences)

Generate the initial teaching prompt:"""
    
    def generate_hint(self, current_step: str, student_progress: float,
                     misconceptions: List[str] = None) -> str:
        """Generate a helpful hint based on student's progress"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
Generate a helpful hint for a student who is working on: {current_step}

Student Progress: {student_progress:.1%}
Misconceptions: {misconceptions or 'None identified'}

The hint should:
1. Not give away the answer
2. Guide toward the next reasoning step
3. Address any misconceptions gently
4. Be encouraging

Keep it concise (1-2 sentences).
"""}
        ]
        
        try:
            return self._call_gpt(messages, temperature=0.1, max_tokens=100)
        except Exception as e:
            logger.error(f"Failed to generate hint: {e}")
            return "Think about what information the image provides that could help answer this question."
    
    def ask_for_clarification(self, unclear_points: List[str],
                            context: List[Dict[str, str]]) -> str:
        """Ask for clarification on unclear points"""
        context_str = "\n".join([
            f"{turn['role'].title()}: {turn['content']}"
            for turn in context[-3:]
        ])
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
The student's response has some unclear points. Generate a clarification request.

Unclear aspects:
{chr(10).join(f'- {point}' for point in unclear_points)}

Recent conversation:
{context_str}

Ask for clarification in a friendly, specific way that helps the student express their thinking more clearly.
"""}
        ]
        
        try:
            return self._call_gpt(messages, temperature=0.7)
        except Exception as e:
            logger.error(f"Failed to generate clarification: {e}")
            return "Could you explain your reasoning a bit more? I want to make sure I understand your thinking."
    
    def provide_encouragement(self, progress: float, strengths: List[str]) -> str:
        """Provide encouraging feedback"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
Generate an encouraging message for a student.

Progress: {progress:.1%}
Strengths observed: {', '.join(strengths) if strengths else 'Good effort'}

Be specific about what they did well and motivate them to continue.
Keep it brief and genuine.
"""}
        ]
        
        try:
            return self._call_gpt(messages, temperature=0.2, max_tokens=150)
        except Exception as e:
            logger.error(f"Failed to generate encouragement: {e}")
            return "Good thinking! You're making progress. Keep going!"
    
    def scaffold_next_step(self, completed_steps: List[str],
                          remaining_steps: List[str]) -> str:
        """Provide scaffolding for the next reasoning step"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
Help scaffold the next reasoning step.

Completed steps:
{chr(10).join(f'✓ {step}' for step in completed_steps)}

Next steps needed:
{chr(10).join(f'- {step}' for step in remaining_steps[:2])}

Generate a prompt that:
1. Acknowledges what's been done
2. Guides toward the next step without giving it away
3. Provides structure for thinking

Keep it concise and actionable.
"""}
        ]
        
        try:
            return self._call_gpt(messages, temperature=0.6)
        except Exception as e:
            logger.error(f"Failed to generate scaffolding: {e}")
            return f"Good work so far. Now, what's the next logical step?"
    
    def provide_final_feedback(self, dialogue_history: List[Dict[str, Any]],
                         final_answer: str, correct_answer: Any,
                         reasoning_quality: float,
                         original_question: str) -> str:
        """Provide comprehensive final feedback"""
        turns = len([d for d in dialogue_history if d['role'] == 'agent'])
        
        # 提取agent的答案内容
        if 'answer_question(answer="' in final_answer:
            import re
            match = re.search(r'answer_question\(answer="([^"]+)"', final_answer)
            if match:
                agent_answer = match.group(1)
        else:
            agent_answer = final_answer
        
        
        is_correct = self._check_answer_correctness(agent_answer, correct_answer)
        
        messages = [
            {"role": "system", "content": "You are a teaching assistant guiding students through visual reasoning. Never reveal the correct answer."},
            {"role": "user", "content": f"""

    Question: {original_question}
    Ground Truth Answer: {correct_answer}
    Agent's Response: {agent_answer}

    Analysis: The agent's answer is {'CORRECT' if is_correct else 'INCORRECT'}.

    RULE:
    If CORRECT: Reply only "Excellent work! Your answer is correct."
    If INCORRECT: Guide them to reconsider without revealing the answer..
    """}
        ]
        
        try:
            return self._call_gpt(messages, temperature=0.3, max_tokens=100)
        except Exception as e:
            logger.error(f"Failed to generate final feedback: {e}")
            if is_correct:
                return "Excellent work! Your answer is correct."
            else:
                return "Let's think about this again. Take another look at the image."
    
    def continue_dialogue(self, current_state: Dict[str, Any],
                         strategy: Any) -> str:
        """Continue the dialogue based on current state"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
Continue the teaching dialogue.

Current State:
- Progress: {current_state.get('progress', 0):.1%}
- Reasoning Quality: {current_state.get('reasoning_quality', 0):.1%}
- Issues: {current_state.get('issues', 'None')}
- Teaching Strategy: {strategy.name}

Generate the next teacher response that:
1. Maintains the teaching strategy
2. Addresses current needs
3. Moves the conversation forward
4. Stays focused on the task

Be natural and conversational.
"""}
        ]
        
        try:
            return self._call_gpt(messages, temperature=0.7)
        except Exception as e:
            logger.error(f"Failed to continue dialogue: {e}")
            return "Let's think about this step by step. What else can you observe?"
    
    def _check_answer_correctness(self, student_answer: str, 
                                correct_answer: Union[str, int, float, bool]) -> bool:
        """Check if student's answer matches correct answer"""
        # Convert to strings for comparison
        student_str = str(student_answer).lower().strip()
        correct_str = str(correct_answer).lower().strip()
        
        # Direct match
        if student_str == correct_str:
            return True
        
        if correct_str in ['yes', 'no']:
            if student_str.startswith(correct_str):
                return True
            # 检查句子中的明确否定/肯定
            if correct_str == 'no' and any(phrase in student_str for phrase in ['is not', "isn't", 'not the']):
                return True
            if correct_str == 'yes' and any(phrase in student_str for phrase in ['is the', 'it is', 'yes,']):
                return True
        
        
        # Numeric comparison with tolerance
        try:
            student_num = float(re.search(r'[-+]?\d*\.?\d+', student_str).group())
            correct_num = float(correct_str)
            return abs(student_num - correct_num) < 0.01
        except:
            pass
        
        # Boolean variations
        bool_true = {'yes', 'true', 'correct', '1'}
        bool_false = {'no', 'false', 'incorrect', '0'}
        
        if correct_str in bool_true and student_str in bool_true:
            return True
        if correct_str in bool_false and student_str in bool_false:
            return True
        
        return False
    
    def _get_fallback_initial_prompt(self, task_info: Dict[str, Any], strategy: Any) -> str:
        """Fallback prompt when GPT fails"""
        task_type = task_info['type']
        question = task_info['question']
        
        # Task-specific fallbacks
        fallbacks = {
            'figureqa': {
                'socratic': "Look at this chart carefully. What information can you extract from it?",
                'direct': f"Let's analyze this chart to answer: {question}",
                'scaffolding': "First, identify what type of chart this is and what it's showing."
            },
            'chartqa': {
                'socratic': "What does this chart tell us? What patterns do you notice?",
                'direct': f"To answer '{question}', we need to read this chart carefully.",
                'scaffolding': "Start by understanding the axes and what each element represents."
            },
            'clevr': {
                'socratic': "What objects do you see in this image? How would you describe them?",
                'direct': f"Count and analyze the objects to answer: {question}",
                'scaffolding': "Let's identify all objects first, then focus on their properties."
            },
            'geometry3k': {
                'socratic': "What geometric relationships do you see in this figure?",
                'direct': f"To solve '{question}', identify the given information in the diagram.",
                'scaffolding': "Mark all the given values on the diagram first."
            },
            'geoqa': {
                'socratic': "What shapes and angles can you identify in this diagram?",
                'direct': f"Let's solve: {question}. Start by listing what we know.",
                'scaffolding': "Begin by labeling all the important points and angles."
            },
            'iconqa': {
                'socratic': "What do you see in this picture?",
                'direct': f"Look at the image to answer: {question}",
                'scaffolding': "Let's examine each part of the image carefully."
            },
            'scienceqa': {
                'socratic': "What scientific concept does this relate to?",
                'direct': f"To answer '{question}', think about the scientific principles involved.",
                'scaffolding': "First, identify what scientific topic this question is about."
            },
            'mathvista': {
                'socratic': "What mathematical relationships do you see here?",
                'direct': f"Let's solve: {question}. What mathematical concepts apply?",
                'scaffolding': "Start by identifying the mathematical elements in the image."
            },
            'olympiadbench': {
                'socratic': "What's your initial approach to this problem?",
                'direct': f"This is a challenging problem: {question}. Let's think systematically.",
                'scaffolding': "Break down this problem into smaller, manageable parts."
            }
        }
        
        # Get task and strategy specific fallback
        if task_type in fallbacks and strategy.name in fallbacks[task_type]:
            return fallbacks[task_type][strategy.name]
        
        # Generic fallback
        return f"Let's work on this {task_type} problem: {question}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            'total_api_calls': self.api_calls,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.cache_hits / max(1, self.api_calls + self.cache_hits),
            'cache_size': len(self.response_cache)
        }
    
    def clear_cache(self):
        """Clear response cache"""
        self.response_cache.clear()
        logger.info("GPT Teacher cache cleared")