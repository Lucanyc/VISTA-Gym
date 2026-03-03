# vlm_gym/environments/gpt_environment/prompts/teaching_prompts.py

"""
Teaching prompt templates for GPT-as-Environment
These templates provide instruction frameworks for GPT to generate contextual responses
"""

from typing import Dict, List, Any, Optional
import logging

class TeachingPrompts:
    """
    Provides instruction templates that guide GPT's teaching behavior
    Templates define HOW to teach, not WHAT to say
    """
    
    def __init__(self):
        """Initialize teaching instruction templates"""
        self.system_prompts = self._init_system_prompts()
        self.hint_templates = self._init_hint_templates()
        self.feedback_templates = self._init_feedback_templates()
        self.clarification_templates = self._init_clarification_templates()
        self.encouragement_templates = self._init_encouragement_templates()
        self.scaffolding_templates = self._init_scaffolding_templates()
    
    def _init_system_prompts(self) -> Dict[str, str]:
        """System prompts that define GPT's teaching persona"""
        return {
            'base': """You are an expert visual reasoning tutor. Your role is to guide students through problems using thoughtful questions and hints, never giving direct answers unless absolutely necessary.

Teaching principles:
1. Use the Socratic method - ask questions that lead to discovery
2. Build on what the student already knows
3. Identify and gently address misconceptions
4. Encourage effort and celebrate progress
5. Adapt complexity to student's level
6. Focus on developing reasoning skills

Always maintain a supportive, patient tone.""",
            
            'socratic': """You are a Socratic teacher for visual reasoning tasks. 

Your approach:
- Ask thought-provoking questions instead of making statements
- Guide students to discover answers themselves
- When they make errors, ask questions that help them see the mistake
- Never directly point out what's wrong; let them realize it
- Use "What do you notice about...?" and "How did you determine...?" frequently""",
            
            'scaffolding': """You are a supportive teacher who provides structured guidance.

Your approach:
- Break complex problems into smaller, manageable steps
- Provide a framework for thinking about the problem
- Give more support initially, then gradually reduce it
- Check understanding at each step before moving forward
- If student struggles, add more structure; if they excel, remove scaffolds""",
            
            'direct': """You are a clear, direct instructor for students who need explicit guidance.

Your approach:
- Provide clear, step-by-step instructions when needed
- Be explicit about what to look for and how to proceed
- Still ask for student input and check understanding
- Use direct teaching when student is stuck after multiple attempts
- Gradually transition to less direct support as student improves"""
        }
    
    def _init_hint_templates(self) -> Dict[str, str]:
        """Templates for generating contextual hints"""
        return {
            'general': """Student's current answer: {student_answer}
Task type: {task_type}
What they're struggling with: {difficulty}
Number of attempts so far: {attempts}

Generate a hint that:
1. Doesn't reveal the answer directly
2. Points their attention to something they may have overlooked
3. Is appropriate for attempt #{attempts} (be progressively more helpful)
4. Relates specifically to their difficulty: {difficulty}

Generate the hint:""",
            
            'observation_focus': """The student needs help with observation.
Their current observation: "{student_observation}"
What they're missing: {missing_element}
Task context: {task_context}

Create a hint that:
1. Guides them to look more carefully at {missing_element}
2. Uses questions like "What do you notice about...?" or "Have you examined...?"
3. Doesn't state what they should see, but where to look
4. Is encouraging and suggests they're capable of seeing it

Generate the hint:""",
            
            'reasoning_support': """The student needs help with reasoning.
Their current reasoning: "{student_reasoning}"
The flaw in their logic: {reasoning_flaw}
Expected reasoning path: {expected_path}

Create a hint that:
1. Helps them recognize the logical issue without stating it
2. Asks questions that reveal the flaw in their thinking
3. Suggests they reconsider specific assumptions
4. Guides toward the correct reasoning path

Generate the hint:""",
            
            'calculation_help': """The student made a calculation error.
Their calculation: {student_calculation}
The error type: {error_type}
Context: {calculation_context}

Create a hint that:
1. Suggests they double-check their arithmetic
2. Points to the specific step where the error might be
3. Doesn't give the correct answer
4. Encourages careful verification

Generate the hint:""",
            
            'comparison_guidance': """The student needs help comparing values.
What they're comparing: {items_to_compare}
Their current conclusion: "{student_conclusion}"
The issue: {comparison_issue}

Create a hint that:
1. Guides them to compare more systematically
2. Suggests a method for comparison without doing it for them
3. Helps them organize their thinking
4. Uses questions to guide discovery

Generate the hint:"""
        }
    
    def _init_feedback_templates(self) -> Dict[str, str]:
        """Templates for generating feedback"""
        return {
            'success': """The student correctly solved the problem!
Their answer: {student_answer}
Key insights they showed: {key_insights}
Number of attempts: {attempts}
Reasoning quality: {reasoning_quality}

Generate encouraging feedback that:
1. Celebrates their success appropriately
2. Highlights what they did well specifically
3. Reinforces the correct reasoning approach
4. Is proportional to the effort (more attempts = more praise for persistence)

Generate the feedback:""",
            
            'partial_success': """The student partially solved the problem.
What they got right: {correct_parts}
What they got wrong: {incorrect_parts}
Their reasoning: {student_reasoning}

Generate feedback that:
1. First acknowledges what they did correctly
2. Gently indicates there's more to consider
3. Doesn't directly state what's wrong
4. Encourages them to refine their answer
5. Maintains their confidence

Generate the feedback:""",
            
            'incorrect_attempt': """The student's answer is incorrect.
Their answer: {student_answer}
Correct answer: {correct_answer}
The main issue: {main_issue}
Attempt number: {attempt}

Generate feedback that:
1. Acknowledges their effort
2. Doesn't reveal the correct answer
3. Provides guidance based on attempt number (more help if many attempts)
4. Addresses the main issue indirectly
5. Keeps them motivated to continue

Generate the feedback:""",
            
            'final_explanation': """Time to explain the correct solution.
Student's final answer: {student_answer}
Correct answer: {correct_answer}
Key concepts involved: {key_concepts}
Student's journey: {learning_journey}

Generate a final explanation that:
1. Clearly explains the correct answer and reasoning
2. Shows where the student's thinking diverged
3. Reinforces key concepts
4. Celebrates what they learned in the process
5. Connects to similar problems they might encounter

Generate the explanation:"""
        }
    
    def _init_clarification_templates(self) -> Dict[str, str]:
        """Templates for requesting clarification"""
        return {
            'unclear_response': """The student's response is unclear.
What they said: "{student_response}"
What's unclear: {unclear_aspects}
Context of discussion: {context}

Generate a clarification request that:
1. Politely indicates you need more information
2. Asks specific questions about the unclear parts
3. Doesn't make them feel bad about being unclear
4. Guides them to express their thinking more precisely

Generate the clarification request:""",
            
            'ambiguous_answer': """The student gave an ambiguous answer.
Their answer: "{student_answer}"
Possible interpretations: {interpretations}
What needs clarification: {clarification_needed}

Generate a request that:
1. Acknowledges their response
2. Explains why clarification is needed (without criticism)
3. Asks them to be more specific
4. Provides structure for their clarification if needed

Generate the clarification request:""",
            
            'incomplete_reasoning': """The student's reasoning is incomplete.
What they explained: "{partial_reasoning}"
What's missing: {missing_steps}
The gap in logic: {logic_gap}

Generate a request that:
1. Acknowledges what they've explained so far
2. Points out there seems to be a jump in reasoning
3. Asks them to fill in the missing steps
4. Uses questions to guide them to complete their thought

Generate the clarification request:"""
        }
    
    def _init_encouragement_templates(self) -> Dict[str, str]:
        """Templates for providing encouragement"""
        return {
            'progress_made': """The student is making progress.
Recent achievement: {achievement}
Current challenge: {current_challenge}
Progress level: {progress_level}
Effort shown: {effort_level}

Generate encouragement that:
1. Specifically acknowledges their achievement
2. Is genuine and proportional to the progress
3. Motivates them to tackle the current challenge
4. Reinforces their capability
5. Is appropriate to their effort level

Generate the encouragement:""",
            
            'persistence_needed': """The student is struggling but trying.
Number of attempts: {attempts}
Frustration level: {frustration_level}
Specific struggle: {struggle_point}
Signs of effort: {effort_signs}

Generate encouragement that:
1. Acknowledges the difficulty
2. Praises their persistence
3. Normalizes struggle as part of learning
4. Offers hope without false promises
5. Suggests a slightly different approach if needed

Generate the encouragement:""",
            
            'breakthrough_moment': """The student just had an insight!
The insight: "{student_insight}"
What led to it: {catalyst}
Significance: {why_important}

Generate encouragement that:
1. Celebrates this "aha" moment enthusiastically
2. Reinforces what led to the breakthrough
3. Connects this insight to the bigger picture
4. Encourages them to apply this thinking
5. Makes them feel proud of their discovery

Generate the encouragement:"""
        }
    
    def _init_scaffolding_templates(self) -> Dict[str, str]:
        """Templates for providing structured support"""
        return {
            'problem_breakdown': """The student needs help breaking down the problem.
The complex problem: {problem_description}
Student's current understanding: {current_understanding}
Task type: {task_type}

Create a scaffolded breakdown that:
1. Divides the problem into 3-5 manageable steps
2. Presents the first step as a question
3. Shows the structure without solving anything
4. Uses the student's current understanding as a starting point
5. Is appropriate for {task_type} problems

Generate the scaffolded guidance:""",
            
            'step_by_step': """Guide the student through the next step.
Current step completed: {completed_step}
Next logical step: {next_step}
Student's approach so far: {approach}

Create guidance that:
1. Acknowledges completion of {completed_step}
2. Naturally leads to the next step without stating it
3. Asks a question that prompts thinking about {next_step}
4. Maintains momentum from their approach
5. Provides just enough structure to proceed

Generate the step guidance:""",
            
            'framework_provision': """Provide a thinking framework.
Problem type: {problem_type}
Student's skill level: {skill_level}
Key concepts involved: {concepts}

Create a framework that:
1. Gives a general approach for {problem_type} problems
2. Is appropriate for {skill_level} level
3. Emphasizes the key concepts: {concepts}
4. Provides structure without specific answers
5. Can be applied to similar problems

Generate the framework:"""
        }
    
    def get_hint_template(self, hint_type: str = 'general') -> str:
        """Get hint generation template"""
        return self.hint_templates.get(hint_type, self.hint_templates['general'])
    
    def get_feedback_template(self, feedback_type: str) -> str:
        """Get feedback generation template"""
        return self.feedback_templates.get(feedback_type, self.feedback_templates['incorrect_attempt'])
    
    def get_clarification_template(self, clarification_type: str = 'unclear_response') -> str:
        """Get clarification request template"""
        return self.clarification_templates.get(clarification_type, self.clarification_templates['unclear_response'])
    
    def get_encouragement_template(self, situation: str = 'progress_made') -> str:
        """Get encouragement template"""
        return self.encouragement_templates.get(situation, self.encouragement_templates['progress_made'])
    
    def get_scaffolding_template(self, scaffold_type: str = 'problem_breakdown') -> str:
        """Get scaffolding template"""
        return self.scaffolding_templates.get(scaffold_type, self.scaffolding_templates['problem_breakdown'])
    
    def get_system_prompt(self, teaching_style: str = 'base') -> str:
        """Get system prompt for teaching style"""
        return self.system_prompts.get(teaching_style, self.system_prompts['base'])
    
    def format_template(self, template: str, context: Dict[str, Any]) -> str:
        """
        Safely format template with context
        
        Args:
            template: Template string with {placeholders}
            context: Dictionary of values to fill
            
        Returns:
            Formatted template ready for GPT
        """
        try:
            # Only format placeholders that exist in context
            # This prevents KeyError if some context is missing
            import re
            
            def replace_placeholder(match):
                key = match.group(1)
                return str(context.get(key, match.group(0)))
            
            # Find all {placeholder} patterns and replace safely
            formatted = re.sub(r'\{(\w+)\}', replace_placeholder, template)
            return formatted
            
        except Exception as e:
            logger.error(f"Error formatting template: {e}")
            return template