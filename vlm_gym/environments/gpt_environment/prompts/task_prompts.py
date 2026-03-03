# vlm_gym/environments/gpt_environment/prompts/task_prompts.py

"""
Unified prompt templates for GPT-as-Environment
Provides instruction frameworks that guide GPT's teaching behavior
"""

from typing import Dict, List, Any, Optional


class TaskPrompts:
    """
    Unified prompt system for GPT-based teaching environment.
    All prompts are instruction templates that tell GPT HOW to teach,
    not WHAT to say. GPT generates contextual responses based on these instructions.
    """
    
    def __init__(self):
        """Initialize all prompt templates"""
        # System-level prompts define GPT's role
        self.system_prompts = self._init_system_prompts()
        
        # Task descriptions help GPT understand different task types
        self.task_descriptions = self._init_task_descriptions()
        
        # Teaching templates for different scenarios
        self.teaching_templates = self._init_teaching_templates()
        
        # Evaluation templates for assessment and feedback
        self.evaluation_templates = self._init_evaluation_templates()
    
    def _init_system_prompts(self) -> Dict[str, str]:
        """System prompts that define GPT's teaching persona and approach"""
        return {
            'base': """You are an expert visual reasoning tutor helping students learn through guided discovery. 

Core principles:
1. Never give direct answers unless the student has genuinely struggled through multiple attempts
2. Use questions to guide thinking rather than statements
3. Build on what the student already knows
4. Identify and gently address misconceptions without directly pointing out errors
5. Provide encouragement proportional to effort
6. Focus on developing reasoning skills, not just getting correct answers

Your responses should be:
- Natural and conversational
- Typically 2-4 sentences
- Focused on one idea at a time
- Encouraging but not overly effusive

Remember: You're teaching them HOW to think, not WHAT to think.""",

            'socratic': """You are a Socratic teacher for visual reasoning tasks.

Your approach:
- Ask thought-provoking questions that lead to discovery
- When students make errors, ask questions that help them see the issue themselves
- Use "What do you notice about...?" and "How did you determine...?" frequently
- Never directly state what's wrong; guide them to realize it
- Build chains of reasoning through connected questions""",

            'scaffolding': """You are a supportive teacher who provides structured guidance.

Your approach:
- Break complex problems into smaller, manageable steps
- Provide just enough structure to keep students progressing
- Check understanding before moving to the next step
- Increase support if struggling, decrease if succeeding
- Make the thinking process visible and systematic""",

            'adaptive': """You are an adaptive teacher who adjusts to each student's needs.

Your approach:
- Start with open-ended guidance
- Increase structure if the student struggles
- Decrease support as they show understanding
- Match language complexity to their demonstrated level
- Switch strategies based on their response patterns"""
        }
    
    def _init_task_descriptions(self) -> Dict[str, str]:
        """Descriptions of each task type to help GPT understand the domain"""
        return {
            'figureqa': """FigureQA involves analyzing charts (bar, line, pie) to answer yes/no questions.
Key aspects: color identification, value comparison, finding min/max, reading scales accurately.
Students often confuse colors or misread axes.""",

            'chartqa': """ChartQA requires extracting specific information from complex visualizations.
Key aspects: precise value reading, understanding units, sometimes calculations.
Students often miss labels or make arithmetic errors.""",

            'clevr': """CLEVR involves reasoning about 3D objects with various properties.
Key aspects: systematic counting, filtering by attributes, spatial relationships.
Students often miss objects or get confused by complex logical conditions.""",

            'geometry3k': """Geometry3K contains high school geometry problems.
Key aspects: theorem application, diagram analysis, multi-step calculations.
Students often miss given information or apply wrong formulas.""",

            'geoqa': """GeoQA focuses on geometric problem-solving with diagrams.
Key aspects: angle relationships, shape properties, spatial reasoning.
Students often misinterpret diagrams or skip steps.""",

            'iconqa': """IconQA presents real-world scenes for younger learners.
Key aspects: counting, pattern recognition, everyday reasoning.
Students often overthink simple questions.""",

            'scienceqa': """ScienceQA tests scientific reasoning with visual contexts.
Key aspects: applying principles, cause-effect reasoning, data interpretation.
Students often bring misconceptions or ignore context.""",

            'mathvista': """MathVista combines mathematical concepts with visualizations.
Key aspects: connecting visual and symbolic representations, calculations.
Students often misread notation or graphs.""",

            'olympiadbench': """OlympiadBench contains competition-level problems.
Key aspects: creative approaches, rigorous reasoning, finding key insights.
Students need to think beyond standard methods."""
        }
    
    def _init_teaching_templates(self) -> Dict[str, str]:
        """Templates for various teaching scenarios"""
        return {
            # Initial engagement with a new task
            'initial_prompt': """Student is starting a {task_type} problem.
Question: "{question}"
Student level: {student_level}
Teaching strategy: {strategy}

Create an engaging opening that:
1. Acknowledges the task without giving away approach
2. Asks an open-ended question to gauge their initial understanding  
3. Is appropriate for {student_level} level
4. Matches the {strategy} teaching style
5. Feels natural and conversational

Generate the opening prompt:""",

            # When student needs a hint
            'hint_generation': """Student needs help with: {difficulty}
Their current answer: "{student_answer}"
Task type: {task_type}
Number of previous hints: {hint_count}
Specific struggle: {struggle_point}

Generate a hint that:
1. Addresses their specific difficulty
2. Doesn't reveal the answer or complete next step
3. Is appropriate for hint #{hint_count} (be progressively more helpful)
4. Uses questions or observation prompts when possible
5. Relates to {task_type} common patterns

Generate the hint:""",

            # When student needs encouragement
            'encouragement': """Student situation: {situation}
Recent progress: {progress_description}
Attempts so far: {attempts}
Current challenge: {challenge}
Effort indicators: {effort_signs}

Generate encouragement that:
1. Acknowledges their specific progress/effort
2. Is genuine and proportional to achievement
3. Addresses their current challenge positively
4. Maintains momentum
5. Avoids empty praise

Generate the encouragement:""",

            # When needing clarification
            'clarification_request': """Student's response is unclear.
What they said: "{student_response}"
What's unclear: {unclear_aspects}
Context: {recent_context}

Generate a clarification request that:
1. Politely indicates confusion
2. Asks specific questions about unclear parts
3. Doesn't make them feel bad
4. Guides toward clearer expression
5. Maintains conversational flow

Generate the request:""",

            # For scaffolding support
            'scaffolding': """Student needs structured help.
Problem: {problem_description}
Current understanding: {student_understanding}
Next logical step: {suggested_next_step}
Task type: {task_type}

Create scaffolding that:
1. Breaks down the current challenge
2. Provides structure without solving
3. Builds on their current understanding
4. Asks guiding questions
5. Is appropriate for {task_type} problems

Generate the scaffolding:""",

            # For addressing errors
            'error_addressing': """Student made an error.
Their response: "{student_response}"
Error type: {error_type}
Correct direction: {correct_direction}
Teaching approach: {approach}

Address the error by:
1. Not directly stating what's wrong
2. Asking questions that lead to self-discovery
3. Focusing on {error_type} specifically
4. Using {approach} teaching style
5. Maintaining student confidence

Generate the response:""",

            # For continuing dialogue
            'continue_dialogue': """Continue the teaching dialogue.
Current state: {dialogue_state}
Student progress: {progress_level}
Recent exchange: {recent_exchange}
Teaching strategy: {strategy}

Generate next response that:
1. Builds on the recent exchange
2. Moves toward the learning goal
3. Maintains {strategy} approach
4. Addresses current needs
5. Keeps natural conversation flow

Generate the continuation:"""
        }
    
    def _init_evaluation_templates(self) -> Dict[str, str]:
        """Templates for evaluation and final feedback"""
        return {
            # For analyzing student work
            'response_analysis': """Analyze the student's response.
Task type: {task_type}
Question: "{question}"
Student answer: "{student_answer}"
Correct answer: "{correct_answer}"
Response quality indicators: {quality_indicators}

Analyze:
1. Correctness (exact, partial, incorrect)
2. Reasoning quality (clarity, completeness, logic)
3. Understanding demonstrated
4. Misconceptions present
5. Progress from previous attempts

Provide structured analysis:""",

            # For final success feedback
            'success_feedback': """Student successfully completed the task!
Their journey: {attempt_summary}
Final answer: "{final_answer}"
Key insights shown: {key_insights}
Struggles overcome: {struggles}
Total attempts: {attempts}

Generate final feedback that:
1. Celebrates appropriately for effort level
2. Highlights specific strengths
3. Reinforces correct reasoning
4. Mentions what they learned
5. Encourages transfer to similar problems
6. Is warm but not over-the-top

Generate the success feedback:""",

            # For incorrect completion feedback
            'incorrect_feedback': """Student completed with incorrect answer.
Their answer: "{student_answer}"
Correct answer: "{correct_answer}"
Their reasoning: {reasoning_summary}
What they got right: {correct_elements}
Main issues: {main_issues}

Generate final feedback that:
1. Acknowledges effort and any correct elements
2. Explains the correct solution clearly
3. Shows where reasoning diverged
4. Emphasizes learning achieved
5. Maintains confidence for future attempts
6. Provides key takeaway

Generate the feedback:""",

            # For progress assessment
            'progress_check': """Assess student's current progress.
Task type: {task_type}
Steps completed: {completed_steps}
Current working on: {current_focus}
Quality of work: {quality_assessment}
Time in task: {duration}

Assess:
1. Progress toward solution (percentage)
2. Understanding demonstrated
3. Likely next challenges
4. Appropriate next intervention
5. Engagement level

Provide assessment:"""
        }
    
    # Core methods for getting prompts
    
    def get_system_prompt(self, teaching_style: str = 'adaptive') -> str:
        """Get the system prompt for GPT's teaching persona"""
        return self.system_prompts.get(teaching_style, self.system_prompts['adaptive'])
    
    def get_task_description(self, task_type: str) -> str:
        """Get description of a task type"""
        return self.task_descriptions.get(
            task_type, 
            "Visual reasoning task requiring careful analysis."
        )
    
    def build_initial_prompt(self, task_type: str, question: str, 
                           student_level: str, strategy: str) -> str:
        """Build the initial teaching prompt for a new task"""
        template = self.teaching_templates['initial_prompt']
        task_desc = self.get_task_description(task_type)
        
        # Combine template with task description for context
        full_prompt = f"""Task context: {task_desc}

{template}"""
        
        return self._format_template(full_prompt, {
            'task_type': task_type,
            'question': question,
            'student_level': student_level,
            'strategy': strategy
        })
    
    def build_hint_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for generating a hint"""
        template = self.teaching_templates['hint_generation']
        return self._format_template(template, context)
    
    def build_encouragement_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for generating encouragement"""
        template = self.teaching_templates['encouragement']
        return self._format_template(template, context)
    
    def build_clarification_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for requesting clarification"""
        template = self.teaching_templates['clarification_request']
        return self._format_template(template, context)
    
    def build_scaffolding_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for providing scaffolding"""
        template = self.teaching_templates['scaffolding']
        return self._format_template(template, context)
    
    def build_error_handling_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for addressing errors"""
        template = self.teaching_templates['error_addressing']
        return self._format_template(template, context)
    
    def build_continuation_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for continuing dialogue"""
        template = self.teaching_templates['continue_dialogue']
        return self._format_template(template, context)
    
    def build_analysis_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for analyzing student response"""
        template = self.evaluation_templates['response_analysis']
        return self._format_template(template, context)
    
    def build_feedback_prompt(self, success: bool, context: Dict[str, Any]) -> str:
        """Build prompt for final feedback"""
        template = (self.evaluation_templates['success_feedback'] if success 
                   else self.evaluation_templates['incorrect_feedback'])
        return self._format_template(template, context)
    
    def build_progress_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for progress assessment"""
        template = self.evaluation_templates['progress_check']
        return self._format_template(template, context)
    
    def _format_template(self, template: str, context: Dict[str, Any]) -> str:
        """
        Safely format template with context values
        
        Args:
            template: Template string with {placeholders}
            context: Dictionary of values to fill
            
        Returns:
            Formatted template ready for GPT
        """
        import re
        
        def replace_placeholder(match):
            key = match.group(1)
            value = context.get(key, f"<{key}>")  # Clear indicator if missing
            return str(value)
        
        # Replace all {placeholder} patterns
        formatted = re.sub(r'\{(\w+)\}', replace_placeholder, template)
        return formatted
    
    def get_common_errors(self, task_type: str) -> List[str]:
        """Get common errors for a task type"""
        errors_by_type = {
            'figureqa': [
                "Confusing which color represents which data",
                "Misreading axis scales or units",
                "Not checking all values for min/max questions"
            ],
            'chartqa': [
                "Missing units in calculations",
                "Arithmetic errors",
                "Misreading labels"
            ],
            'clevr': [
                "Miscounting due to unsystematic approach",
                "Attribute confusion",
                "Missing partially hidden objects"
            ],
            'geometry3k': [
                "Wrong theorem application",
                "Missing given information",
                "Calculation errors"
            ],
            # ... other task types
        }
        return errors_by_type.get(task_type, ["Hasty conclusions", "Missing details"])
    
    def get_teaching_focus(self, task_type: str, challenge: str) -> str:
        """Get specific teaching focus for a task and challenge"""
        focuses = {
            'figureqa': {
                'observation': "Guide careful examination of chart elements and legends",
                'comparison': "Help organize systematic value comparison",
                'confusion': "Clarify which visual element represents which data"
            },
            'chartqa': {
                'observation': "Ensure all labels and units are identified",
                'calculation': "Guide through step-by-step calculation process",
                'extraction': "Help locate specific data points accurately"
            },
            # ... other task types
        }
        
        task_focuses = focuses.get(task_type, {})
        for key in ['observation', 'comparison', 'calculation', 'confusion']:
            if key in challenge.lower():
                return task_focuses.get(key, "Guide appropriately based on the challenge")
        
        return "Provide appropriate guidance for the specific challenge"