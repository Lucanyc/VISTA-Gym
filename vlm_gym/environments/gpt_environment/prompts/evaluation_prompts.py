# vlm_gym/environments/gpt_environment/prompts/evaluation_prompts.py

"""
Evaluation and feedback prompt templates
"""

from typing import Dict, List, Optional
import random


class EvaluationPrompts:
    """
    Provides prompts for evaluation, feedback, and assessment scenarios
    """
    
    def __init__(self):
        """Initialize evaluation prompt templates"""
        self.final_feedback_templates = self._init_final_feedback_templates()
        self.error_analysis_templates = self._init_error_analysis_templates()
        self.success_feedback_templates = self._init_success_feedback_templates()
        self.partial_credit_templates = self._init_partial_credit_templates()
        self.summary_templates = self._init_summary_templates()
        self.improvement_templates = self._init_improvement_templates()
    
    def _init_final_feedback_templates(self) -> Dict[str, List[str]]:
        """Initialize final feedback templates"""
        return {
            'complete_success': [
                "Excellent work! You correctly solved the problem by {key_approach}. Your {strength} was particularly impressive.",
                "Perfect! You demonstrated clear understanding of {concept} and applied it correctly. Well done!",
                "Outstanding! Your systematic approach to {task_type} problems shows great understanding.",
                "Congratulations! You not only got the right answer but also showed excellent {skill}.",
                "Superb work! Your {reasoning_quality} reasoning led you to the correct solution."
            ],
            
            'success_with_struggles': [
                "Well done! Despite initial challenges with {difficulty}, you persevered and found the correct answer.",
                "Great job! You corrected your {initial_error} and arrived at the right solution. That shows good self-monitoring.",
                "Excellent recovery! After {struggle}, you adjusted your approach and succeeded.",
                "Good work! Your ability to recognize and fix {mistake} led to the correct answer.",
                "Nice job! Even though {challenge} was tricky, you worked through it successfully."
            ],
            
            'incorrect_but_good_effort': [
                "Good effort! While the answer isn't correct, your approach to {positive_aspect} was sound. The issue was with {error}.",
                "Nice try! You showed good thinking about {strength}, but {weakness} led to an incorrect answer. Let's review {concept}.",
                "I appreciate your systematic approach. The error occurred in {error_point}. Next time, remember to {improvement}.",
                "You're on the right track with {good_part}. To get the correct answer, you need to {correction}.",
                "Good reasoning in parts! Your {strength} was correct, but {error} needs adjustment."
            ],
            
            'incorrect_fundamental': [
                "Let's work on understanding {fundamental_concept}. The correct answer is {answer} because {explanation}.",
                "This type of problem requires {key_skill}. The answer is {answer}. Let me explain the approach: {explanation}.",
                "Don't worry - {task_type} problems can be challenging. The answer is {answer}. Key insight: {insight}.",
                "Learning opportunity! For {problem_type}, remember {key_principle}. The correct answer is {answer}.",
                "Let's review this together. The answer is {answer}. The main concept to understand is {concept}."
            ]
        }
    
    def _init_error_analysis_templates(self) -> Dict[str, List[str]]:
        """Initialize error analysis templates"""
        return {
            'calculation_error': [
                "Your approach was correct, but there was a calculation error in {location}. Double-check arithmetic next time.",
                "Small calculation mistake in {step}. Remember to verify each computation.",
                "The math error occurred when {operation}. Always recheck numerical work.",
                "Arithmetic slip in {calculation}. The method was right, just recalculate carefully.",
                "Computational error: {specific_error}. Your reasoning was sound otherwise."
            ],
            
            'conceptual_error': [
                "There's a misunderstanding about {concept}. Remember that {correct_explanation}.",
                "The error stems from confusion about {topic}. Key point: {clarification}.",
                "Let's clarify {concept}: {explanation}. This is why {consequence}.",
                "Common misconception about {topic}. Actually, {correct_understanding}.",
                "The issue is with understanding {principle}. Here's how it really works: {explanation}."
            ],
            
            'reading_error': [
                "Careful reading needed: you interpreted {misread} when it actually shows {correct}.",
                "The {element} was misread. It actually indicates {correct_reading}.",
                "Check the {source} again - you saw {wrong} but it's {right}.",
                "Observation error: the {item} is {actual}, not {perceived}.",
                "Misinterpretation of {component}. Look more carefully at {detail}."
            ],
            
            'logical_error': [
                "Logic gap between {premise} and {conclusion}. The connection needs {missing_link}.",
                "The reasoning breaks down at {point}. Consider {alternative_logic}.",
                "Your conclusion doesn't follow from {evidence}. Think about {correct_reasoning}.",
                "Logical error: {fallacy}. Instead, {correct_logic}.",
                "The inference from {observation} to {conclusion} needs {intermediate_step}."
            ],
            
            'incomplete_error': [
                "You addressed {completed} but missed {missing}. Complete solutions need both.",
                "Partial answer: you got {correct_part} right but didn't consider {missing_part}.",
                "Incomplete analysis - you need to also examine {overlooked_aspect}.",
                "You stopped at {stopping_point}. Continue to find {remaining}.",
                "Good start with {initial}, but {final} still needs to be determined."
            ]
        }
    
    def _init_success_feedback_templates(self) -> Dict[str, List[str]]:
        """Initialize success celebration templates"""
        return {
            'quick_success': [
                "Brilliant! You solved that efficiently in just {turns} steps!",
                "Impressive speed and accuracy! Well done!",
                "Quick and correct - excellent problem-solving!",
                "Efficient solution! You went straight to the answer.",
                "Fast and accurate - great job!"
            ],
            
            'methodical_success': [
                "Excellent systematic approach! Your step-by-step method is exemplary.",
                "Thorough and correct! Your attention to detail paid off.",
                "Perfect demonstration of methodical problem-solving!",
                "Your careful analysis led to the right answer. Well done!",
                "Systematic and successful - exactly the right approach!"
            ],
            
            'creative_success': [
                "Creative solution! I like how you approached this differently.",
                "Innovative thinking! Your unique approach worked perfectly.",
                "Original and correct - impressive problem-solving!",
                "Creative insight led to the right answer. Excellent!",
                "Unique approach that worked beautifully. Well done!"
            ],
            
            'persistent_success': [
                "Your persistence paid off! Great job sticking with it.",
                "Determination wins! You didn't give up and found the answer.",
                "Perseverance leads to success - well done!",
                "Great resilience! You worked through the challenges.",
                "Your determination to get it right is admirable!"
            ]
        }
    
    def _init_partial_credit_templates(self) -> Dict[str, List[str]]:
        """Initialize partial credit templates"""
        return {
            'mostly_correct': [
                "Almost perfect! Just a small issue with {minor_error}. Overall excellent work.",
                "Very close! Only {small_mistake} keeps this from being completely correct.",
                "Nearly there! Fix {minor_issue} and you'd have it perfect.",
                "90% correct! Just need to adjust {small_detail}.",
                "Almost got it! Small correction needed in {minor_aspect}."
            ],
            
            'half_correct': [
                "Partially correct. You got {correct_parts} right, but need to work on {incorrect_parts}.",
                "Mixed results: {successes} were good, but {failures} need attention.",
                "Half way there! Strong on {strengths}, needs work on {weaknesses}.",
                "Some correct elements ({correct}), some incorrect ({incorrect}).",
                "Partial understanding shown. Good: {good_parts}. Needs work: {bad_parts}."
            ],
            
            'good_process_wrong_answer': [
                "Good thinking process, but {error} led to wrong answer. Method was sound!",
                "Right approach, wrong execution. The {mistake} threw off your answer.",
                "I like your reasoning! Just fix {error_point} to get the right answer.",
                "Solid methodology, but {slip} caused an error. You're on the right track!",
                "Good problem-solving approach! The {issue} was the only problem."
            ],
            
            'right_answer_poor_reasoning': [
                "Correct answer, but let's strengthen the reasoning. {improvement_suggestion}.",
                "You got it right! To improve, work on {reasoning_aspect}.",
                "Right answer! Next time, be clearer about {explanation_need}.",
                "Correct, but your explanation could be stronger. Consider {suggestion}.",
                "Yes, that's right! For full marks, include {missing_reasoning}."
            ]
        }
    
    def _init_summary_templates(self) -> Dict[str, str]:
        """Initialize problem summary templates"""
        return {
            'task_summary': "This {task_type} problem asked you to {objective}. The key was {key_insight}.",
            'performance_summary': "You completed this in {turns} turns with {quality} reasoning quality.",
            'learning_summary': "Main takeaway: {lesson}. Remember this for similar problems.",
            'skill_summary': "This problem tested {skills}. You showed strength in {strengths}.",
            'strategy_summary': "The {strategy} approach worked {effectiveness} for this problem."
        }
    
    def _init_improvement_templates(self) -> Dict[str, List[str]]:
        """Initialize improvement suggestion templates"""
        return {
            'efficiency': [
                "To solve this faster next time, {efficiency_tip}.",
                "Time-saving tip: {shortcut} can speed up your solution.",
                "For efficiency, try {method} instead of {current_method}.",
                "Quick tip: {time_saver} reduces steps needed.",
                "Streamline by {improvement}."
            ],
            
            'accuracy': [
                "To improve accuracy, always {accuracy_tip}.",
                "Precision tip: {checking_method} helps avoid errors.",
                "For better accuracy, {verification_step}.",
                "Reduce errors by {error_prevention}.",
                "Accuracy boost: {precision_method}."
            ],
            
            'understanding': [
                "Deepen understanding by {study_suggestion}.",
                "To grasp this better, focus on {concept_focus}.",
                "Understanding tip: {conceptual_hint}.",
                "Master this by {learning_method}.",
                "Conceptual clarity: {understanding_tip}."
            ],
            
            'problem_solving': [
                "Problem-solving tip: {strategy_suggestion}.",
                "Approach improvement: {method_enhancement}.",
                "For similar problems, {tactical_advice}.",
                "Strategic insight: {problem_solving_tip}.",
                "Next time, try {alternative_approach}."
            ]
        }
    
    def get_final_feedback(self, success: bool, quality: str = 'complete_success',
                          context: Optional[Dict[str, str]] = None) -> str:
        """
        Get final feedback for task completion
        
        Args:
            success: Whether the task was completed successfully
            quality: Quality category of the completion
            context: Context variables for formatting
            
        Returns:
            Formatted feedback
        """
        if success and quality not in self.final_feedback_templates:
            quality = 'complete_success'
        elif not success and quality not in self.final_feedback_templates:
            quality = 'incorrect_but_good_effort'
            
        templates = self.final_feedback_templates[quality]
        template = random.choice(templates)
        
        if context:
            try:
                return template.format(**context)
            except KeyError:
                pass
                
        return template
    
    def get_error_analysis(self, error_type: str = 'conceptual_error',
                          context: Optional[Dict[str, str]] = None) -> str:
        """Get error analysis feedback"""
        templates = self.error_analysis_templates.get(error_type,
                                                     self.error_analysis_templates['conceptual_error'])
        template = random.choice(templates)
        
        if context:
            try:
                return template.format(**context)
            except KeyError:
                pass
                
        return template
    
    def get_success_feedback(self, success_type: str = 'methodical_success',
                           context: Optional[Dict[str, str]] = None) -> str:
        """Get success celebration feedback"""
        templates = self.success_feedback_templates.get(success_type,
                                                       self.success_feedback_templates['methodical_success'])
        template = random.choice(templates)
        
        if context:
            try:
                return template.format(**context)
            except KeyError:
                pass
                
        return template
    
    def get_partial_credit_feedback(self, credit_type: str = 'mostly_correct',
                                  context: Optional[Dict[str, str]] = None) -> str:
        """Get partial credit feedback"""
        templates = self.partial_credit_templates.get(credit_type,
                                                     self.partial_credit_templates['half_correct'])
        template = random.choice(templates)
        
        if context:
            try:
                return template.format(**context)
            except KeyError:
                pass
                
        return template
    
    def get_summary(self, summary_type: str = 'task_summary',
                   context: Optional[Dict[str, str]] = None) -> str:
        """Get problem summary"""
        template = self.summary_templates.get(summary_type,
                                            self.summary_templates['task_summary'])
        
        if context:
            try:
                return template.format(**context)
            except KeyError:
                pass
                
        return template
    
    def get_improvement_suggestion(self, improvement_area: str = 'problem_solving',
                                 context: Optional[Dict[str, str]] = None) -> str:
        """Get improvement suggestion"""
        templates = self.improvement_templates.get(improvement_area,
                                                 self.improvement_templates['problem_solving'])
        template = random.choice(templates)
        
        if context:
            try:
                return template.format(**context)
            except KeyError:
                pass
                
        return template
    
    def create_comprehensive_feedback(self, components: List[str],
                                    separator: str = "\n\n") -> str:
        """
        Create comprehensive feedback from multiple components
        
        Args:
            components: List of feedback components to combine
            separator: String to separate components
            
        Returns:
            Combined feedback message
        """
        # Filter out empty components
        components = [c for c in components if c and c.strip()]
        return separator.join(components)
    
    def personalize_feedback(self, feedback: str, student_name: Optional[str] = None,
                           encouragement_level: str = 'moderate') -> str:
        """
        Personalize feedback based on student profile
        
        Args:
            feedback: Base feedback message
            student_name: Optional student name
            encouragement_level: Level of encouragement ('minimal', 'moderate', 'high')
            
        Returns:
            Personalized feedback
        """
        if student_name:
            feedback = f"{student_name}, {feedback[0].lower()}{feedback[1:]}"
        
        if encouragement_level == 'high':
            encouragers = ["Keep up the great work!", "You're doing amazing!", 
                         "I'm impressed with your progress!"]
            feedback += f" {random.choice(encouragers)}"
        elif encouragement_level == 'minimal':
            # Keep feedback concise and matter-of-fact
            feedback = feedback.replace("excellent", "good").replace("brilliant", "correct")
            
        return feedback