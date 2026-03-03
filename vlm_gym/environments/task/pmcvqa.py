from typing import Tuple, Dict, Any, List, Optional
import re
import logging

from .vision_qa_task import VisionQATask

logger = logging.getLogger(__name__)


class PMCVQATask(VisionQATask):
    """
    PMC-VQA Medical Visual Question Answering Task
    
    Specialized task for medical image question answering with:
    - Medical terminology support
    - Anatomical structure detection
    - Radiological finding analysis
    - Tool integration for medical image analysis
    """
    
    @classmethod
    def get_task_id(cls) -> str:
        """Get task type ID"""
        return "vlm-gym.pmc-vqa"
    
    def setup(self) -> Tuple[str, Dict[str, Any]]:
        """Setup PMC-VQA specific task configuration"""
        # Call parent setup
        task_goal, task_info = super().setup()
        
        # Add PMC-VQA specific processing
        task_info["medical_domain"] = self._detect_medical_domain()
        task_info["requires_localization"] = self._requires_localization()
        task_info["imaging_modality"] = self._detect_imaging_modality()
        task_info["dataset"] = "pmc_vqa"
        
        # Enhance task goal with medical-specific guidance
        enhanced_goal = task_goal
        
        # Add medical imaging guidance
        enhanced_goal += "\n\n**Medical Image Analysis Guidelines:**"
        enhanced_goal += "\n- Carefully examine the medical image for relevant anatomical structures"
        enhanced_goal += "\n- Look for any abnormalities, lesions, or specific findings mentioned in the question"
        enhanced_goal += "\n- Consider the imaging modality and what it best visualizes"
        
        # Add tool usage hints based on question type
        if task_info["requires_localization"]:
            enhanced_goal += "\n\n**Localization Required:** This question asks about specific locations or regions."
            enhanced_goal += "\n- Use visual grounding tools to identify and mark relevant areas"
            enhanced_goal += "\n- Pay attention to anatomical landmarks"
        
        # Add domain-specific guidance
        domain_guidance = self._get_domain_specific_guidance(task_info["medical_domain"])
        if domain_guidance:
            enhanced_goal += f"\n\n**Domain-Specific Notes:** {domain_guidance}"
        
        # Add reminder about multiple choice format
        if self.choices:
            enhanced_goal += "\n\n**Important:** Select your answer from the provided choices only."
            enhanced_goal += "\nEnsure your final answer is one of the given options."
        
        return enhanced_goal, task_info
    
    def _detect_medical_domain(self) -> str:
        """Detect the medical specialty domain from question content"""
        if not self.question:
            return "general"
        
        question_lower = self.question.lower()
        
        # Domain keywords mapping
        domain_keywords = {
            "radiology": ["radiological", "x-ray", "ct", "mri", "ultrasound", "scan"],
            "oncology": ["tumor", "cancer", "malignant", "benign", "metastasis", "lesion"],
            "cardiology": ["cardiac", "heart", "coronary", "vessel", "artery", "vein"],
            "neurology": ["brain", "neural", "cerebral", "spinal", "neurological"],
            "pathology": ["pathological", "histology", "tissue", "cell", "biopsy"],
            "mammography": ["breast", "mammography", "mammogram", "calcification"],
            "nuclear_medicine": ["uptake", "tracer", "fdg", "pet", "spect"]
        }
        
        # Check for domain matches
        for domain, keywords in domain_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                return domain
        
        # Check metadata for additional clues
        metadata = self.metadata or {}
        if metadata.get("domain") == "medical":
            return "general_medical"
        
        return "general"
    
    def _requires_localization(self) -> bool:
        """Determine if the question requires spatial localization"""
        if not self.question:
            return False
        
        localization_keywords = [
            "where", "location", "located", "position", "region",
            "area", "site", "which part", "what part", "identify",
            "point", "mark", "show", "find", "localize"
        ]
        
        question_lower = self.question.lower()
        return any(keyword in question_lower for keyword in localization_keywords)
    
    def _detect_imaging_modality(self) -> str:
        """Detect the imaging modality from question or metadata"""
        if not self.question:
            return "unknown"
        
        question_lower = self.question.lower()
        
        # Modality detection patterns
        modalities = {
            "mammography": ["mammography", "mammogram"],
            "ct": ["ct scan", "computed tomography", "ct"],
            "mri": ["mri", "magnetic resonance"],
            "xray": ["x-ray", "xray", "radiograph"],
            "ultrasound": ["ultrasound", "sonography", "echo"],
            "pet": ["pet scan", "pet", "positron emission"],
            "nuclear": ["scintigraphy", "nuclear medicine", "spect"],
            "microscopy": ["microscopy", "histology", "pathology slide"]
        }
        
        for modality, keywords in modalities.items():
            if any(keyword in question_lower for keyword in keywords):
                return modality
        
        # Check for clues in the answer/choices
        all_text = question_lower
        if self.answer:
            all_text += " " + str(self.answer).lower()
        if self.choices:
            all_text += " " + " ".join([str(c).lower() for c in self.choices])
        
        for modality, keywords in modalities.items():
            if any(keyword in all_text for keyword in keywords):
                return modality
        
        return "general_medical_imaging"
    
    def _get_domain_specific_guidance(self, domain: str) -> str:
        """Provide domain-specific guidance for better analysis"""
        guidance_map = {
            "radiology": "Focus on imaging characteristics, density differences, and anatomical structures.",
            "oncology": "Look for masses, irregular borders, enhancement patterns, and size measurements.",
            "mammography": "Pay attention to calcifications, masses, asymmetries, and breast density.",
            "nuclear_medicine": "Examine uptake patterns, intensity levels, and distribution of tracers.",
            "pathology": "Analyze cellular structures, tissue organization, and staining patterns.",
            "cardiology": "Focus on vessel structures, cardiac chambers, and flow patterns.",
            "neurology": "Examine brain regions, grey/white matter differentiation, and any abnormalities."
        }
        
        return guidance_map.get(domain, "")
    
    def check_success(self, action: Any) -> Tuple[bool, str]:
        """
        Check PMC-VQA answer with medical-specific validation
        
        Args:
            action: User's answer
            
        Returns:
            success: Whether the answer is correct
            feedback: Feedback message
        """
        if not action:
            return False, "No answer provided. Please select one of the given choices."
        
        # Clean the answer
        action = str(action).strip()
        
        # Get the correct answer letter from metadata
        correct_letter = self.metadata.get('ground_truth_answer', '').upper()
        
        # Handle letter answer (A, B, C, D)
        if len(action) == 1 and action.upper() in 'ABCD':
            if action.upper() == correct_letter:
                return True, "Correct! Your medical image analysis is accurate."
            else:
                # Provide informative feedback
                if self.choices and 0 <= ord(correct_letter) - ord('A') < len(self.choices):
                    correct_choice = self.choices[ord(correct_letter) - ord('A')]
                    return False, f"Incorrect. The correct answer is {correct_letter}: {correct_choice}"
                return False, f"Incorrect. The correct answer is {correct_letter}."
        
        # Handle full text answer
        if self.choices:
            # Check if the answer matches one of the choices
            action_lower = action.lower()
            for i, choice in enumerate(self.choices):
                if choice.lower() in action_lower or action_lower in choice.lower():
                    choice_letter = chr(ord('A') + i)
                    if choice_letter == correct_letter:
                        return True, "Correct! Your answer matches the expected choice."
                    else:
                        return False, f"Incorrect. You selected option {choice_letter}, but the correct answer is {correct_letter}."
        
        # Handle answer that matches the stored answer field
        if self.answer and action.lower() == str(self.answer).lower():
            return True, "Correct! Your answer is accurate."
        
        # If none of the above, it's incorrect
        return False, f"Incorrect. Please select one of the given choices. The correct answer is {correct_letter}."
    
    def validate(
        self,
        chat_history: List[Dict],
        observation: Any,
        full_history: Optional[List[Any]] = None
    ) -> Tuple[float, bool, str, Dict[str, Any]]:
        """
        Validate PMC-VQA task execution with medical-specific metrics
        """
        # Call parent validation
        reward, done, message, info = super().validate(
            chat_history, observation, full_history
        )
        
        # Add PMC-VQA specific information
        info["medical_domain"] = self._detect_medical_domain()
        info["required_localization"] = self._requires_localization()
        info["imaging_modality"] = self._detect_imaging_modality()
        
        # Track tool usage if available
        if full_history:
            tools_used = self._extract_tools_used(full_history)
            info["tools_used"] = tools_used
            info["tool_count"] = len(tools_used)
            
            # Bonus for appropriate tool usage
            if info.get("required_localization") and "grounding_dino" in tools_used:
                info["appropriate_tool_usage"] = True
        
        return reward, done, message, info
    
    def _extract_tools_used(self, history: List[Any]) -> List[str]:
        """Extract which tools were used during task execution"""
        tools_used = []
        
        for entry in history:
            if isinstance(entry, dict):
                # Check for tool invocation patterns
                if entry.get("tool_name"):
                    tools_used.append(entry["tool_name"])
                elif entry.get("action") == "tool_use":
                    tools_used.append(entry.get("tool", "unknown"))
                elif "grounding_dino" in str(entry).lower():
                    tools_used.append("grounding_dino")
                elif "deepeyes" in str(entry).lower():
                    tools_used.append("deepeyes")
        
        return list(set(tools_used))  # Remove duplicates
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get PMC-VQA specific metrics"""
        metrics = super().get_metrics()
        
        metrics.update({
            "medical_domain": self._detect_medical_domain(),
            "requires_localization": self._requires_localization(),
            "imaging_modality": self._detect_imaging_modality(),
            "question_complexity": self._assess_complexity(),
            "has_ground_truth": self.metadata.get('ground_truth_answer') is not None
        })
        
        return metrics
    
    def _assess_complexity(self) -> str:
        """Assess medical question complexity"""
        if not self.question:
            return "unknown"
        
        question_lower = self.question.lower()
        
        # Complex medical analysis indicators
        complex_indicators = [
            "differential diagnosis", "compare", "explain", "analyze",
            "pathophysiology", "mechanism", "progression", "staging"
        ]
        
        # Moderate complexity indicators
        moderate_indicators = [
            "identify", "describe", "characteristic", "finding",
            "abnormality", "feature", "pattern"
        ]
        
        # Simple identification
        simple_indicators = [
            "what is", "which", "where", "name", "type"
        ]
        
        if any(ind in question_lower for ind in complex_indicators):
            return "high"
        elif any(ind in question_lower for ind in moderate_indicators):
            return "medium"
        elif any(ind in question_lower for ind in simple_indicators):
            return "low"
        else:
            return "medium"
    
    def get_hint(self) -> str:
        """Provide a medical-specific hint for the task"""
        hints = []
        
        # Domain-specific hints
        domain = self._detect_medical_domain()
        if domain == "mammography":
            hints.append("Look for calcifications, masses, or architectural distortions")
        elif domain == "nuclear_medicine":
            hints.append("Focus on areas of increased or decreased tracer uptake")
        elif domain == "radiology":
            hints.append("Compare different tissue densities and anatomical structures")
        
        # Localization hints
        if self._requires_localization():
            hints.append("Use visual grounding to identify the specific region being asked about")
        
        # General hint
        if not hints:
            hints.append("Carefully analyze the medical image and consider all visible findings")
        
        return ". ".join(hints)