"""
VQA-RAD Task implementation for VLM Gym
Handles Medical Visual Question Answering tasks on radiology images
"""

from typing import Tuple, Dict, Any, List, Optional
import re
import json
from pathlib import Path
from collections import Counter

from .vision_qa_task import VisionQATask


class VqaRadTask(VisionQATask):
    """
    VQA-RAD (Visual Question Answering - Radiology) 特定任务
    
    专门处理医学影像（放射科图像）的视觉问答任务，包括：
    - 诊断问题（what is the diagnosis, what condition）
    - 解剖识别（which organ, what structure）
    - 异常检测（is there, are there abnormalities）
    - 成像模态（what type of scan, how was this taken）
    - 病变定位（where is the lesion）
    - 医学验证（is this normal, are there findings）
    - 描述性问题（describe what you see）
    - 测量估计（size, dimension）
    
    支持二元判断（yes/no）和开放式医学答案
    """
    
    # 医学任务类型
    TASK_TYPES = {
        'medical_diagnosis': '医学诊断',
        'anatomical_recognition': '解剖结构识别',
        'abnormality_detection': '异常检测',
        'modality_recognition': '成像模态识别',
        'lesion_localization': '病变定位',
        'medical_verification': '医学验证（Yes/No）',
        'medical_description': '医学描述',
        'measurement_estimation': '测量估计',
        'medical_counting': '医学计数',
        'medical_vqa': '通用医学视觉问答'
    }
    
    # 医学问题类型
    QUESTION_TYPES = {
        'diagnosis': '诊断类（diagnosis, condition）',
        'anatomy': '解剖类（organ, structure）',
        'abnormality': '异常类（lesion, mass, finding）',
        'modality': '模态类（MRI, CT, X-ray）',
        'location': '位置类（where, which side）',
        'yes_no': 'Yes/No判断类',
        'description': '描述类（describe, what is seen）',
        'counting': '计数类（how many）',
        'measurement': '测量类（size, dimension）',
        'general': '通用类'
    }
    
    # 常见医学答案类别
    ANSWER_CATEGORIES = {
        'yes_no': ['yes', 'no'],
        'modalities': ['mri', 'ct', 'x-ray', 'ultrasound', 'pet', 'spect'],
        'anatomical_structures': ['brain', 'lung', 'heart', 'liver', 'kidney', 'spine', 'chest', 'abdomen', 'skull'],
        'abnormalities': ['mass', 'lesion', 'tumor', 'cyst', 'nodule', 'fracture', 'hemorrhage', 'edema', 'effusion'],
        'locations': ['left', 'right', 'bilateral', 'frontal', 'temporal', 'parietal', 'occipital', 'upper', 'lower'],
        'conditions': ['normal', 'abnormal', 'pneumonia', 'cardiomegaly', 'atelectasis', 'consolidation']
    }
    
    # 医学复杂度标记
    MEDICAL_COMPLEXITY_MARKERS = {
        'medical_terms': ['infarct', 'hemorrhage', 'herniation', 'consolidation', 'atelectasis', 'cardiomegaly'],
        'anatomical_terms': ['ventricle', 'lobe', 'cortex', 'mediastinum', 'parenchyma', 'pleura'],
        'differential_markers': [' or ', 'differential', 'versus', 'rule out'],
        'laterality_terms': ['left', 'right', 'bilateral', 'unilateral']
    }
    
    def __init__(self, task_id: str, adapter: Any):
        """
        初始化VQA-RAD任务
        
        Args:
            task_id: 任务ID
            adapter: VQA-RAD数据适配器
        """
        # 调用父类初始化
        super().__init__(task_id, adapter)
        
        # 初始化医学VQA特定属性
        self.task_type = None
        self.question_type = None
        self.answer_type = None  # 'yes_no' or 'open_ended'
        self.expected_answer = None
        self.is_clinical = False
        self.medical_entities = {}
        self.medical_complexity = {}
        self.is_complex = False
        self.is_binary = False
        self.answer_length_category = None
        
        # 获取任务数据
        task_data = adapter.get_task_data(task_id)
        self.dataset_name = task_data.get('dataset', 'vqa_rad')
        self.task_type = task_data.get('task', 'medical_vqa')
        self.choices = task_data.get('choices', None)
    
    @classmethod
    def get_task_id(cls) -> str:
        """获取任务类型ID"""
        return "vlm-gym.vqa-rad"
    
    def setup(self) -> Tuple[str, Dict[str, Any]]:
        """设置VQA-RAD特定的任务"""
        # 调用父类setup
        task_goal, task_info = super().setup()
        
        # 从metadata中提取医学VQA特定信息
        self.question_type = self.metadata.get('question_type', 'general')
        self.answer_type = self.metadata.get('answer_type', 'open_ended')
        self.expected_answer = self.answer.lower() if self.answer else None
        self.is_binary = self.answer_type == 'yes_no' or self.expected_answer in ['yes', 'no']
        self.is_clinical = self.metadata.get('is_clinical', False)
        
        # 获取医学分析信息
        self.question_analysis = self.metadata.get('question_analysis', {})
        self.medical_entities = self.question_analysis.get('entities', {})
        self.medical_complexity = self.question_analysis.get('complexity', {})
        self.is_complex = self.question_analysis.get('is_complex', False)
        
        # 判断答案长度类别
        answer_words = len(self.expected_answer.split()) if self.expected_answer else 0
        if answer_words == 1:
            self.answer_length_category = 'short'
        elif answer_words <= 3:
            self.answer_length_category = 'medium'
        else:
            self.answer_length_category = 'long'
        
        # 分析医学问题特征
        medical_features = self._analyze_medical_features()
        
        # 添加医学VQA特定的信息
        task_info["task_type"] = self.task_type
        task_info["question_type"] = self.question_type
        task_info["answer_type"] = self.answer_type
        task_info["expected_answer"] = self.expected_answer
        task_info["is_binary"] = self.is_binary
        task_info["is_clinical"] = self.is_clinical
        task_info["is_complex"] = self.is_complex
        task_info["answer_length"] = self.answer_length_category
        task_info["medical_features"] = medical_features
        task_info["medical_entities"] = self.medical_entities
        task_info["medical_complexity"] = self.medical_complexity
        task_info["dataset"] = "vqa_rad"
        task_info["medical_domain"] = self.metadata.get('medical_domain', 'radiology')
        task_info["difficulty"] = self._assess_medical_difficulty()
        
        # 修改任务目标以包含医学VQA特定指导
        enhanced_goal = self._enhance_medical_task_goal(task_goal, medical_features)
        
        return enhanced_goal, task_info
    
    def _enhance_medical_task_goal(self, base_goal: str, features: Dict[str, bool]) -> str:
        """增强任务目标描述，添加医学VQA特定的指导"""
        enhanced_parts = [base_goal]
        
        # 添加医学影像理解通用指导
        enhanced_parts.append("\n**Medical Image Understanding:**")
        enhanced_parts.append("1. Carefully observe the medical image (radiology scan)")
        enhanced_parts.append("2. Identify anatomical structures and any abnormalities")
        enhanced_parts.append("3. Consider the imaging modality and view")
        enhanced_parts.append("4. Apply medical knowledge to answer the specific question")
        
        # 根据任务类型添加特定指导
        if self.task_type == 'medical_diagnosis':
            enhanced_parts.append("\n**Diagnosis Task:**")
            enhanced_parts.append("- Identify the medical condition or diagnosis")
            enhanced_parts.append("- Use proper medical terminology")
            enhanced_parts.append("- Be specific about the finding")
        
        elif self.task_type == 'anatomical_recognition':
            enhanced_parts.append("\n**Anatomical Recognition Task:**")
            enhanced_parts.append("- Identify the anatomical structure or organ system")
            enhanced_parts.append("- Use correct anatomical terminology")
            enhanced_parts.append("- Consider the imaging plane and orientation")
        
        elif self.task_type == 'abnormality_detection':
            enhanced_parts.append("\n**Abnormality Detection Task:**")
            enhanced_parts.append("- Look for any abnormal findings")
            enhanced_parts.append("- Consider normal vs abnormal appearance")
            enhanced_parts.append("- Be specific about the type of abnormality")
        
        elif self.task_type == 'modality_recognition':
            enhanced_parts.append("\n**Imaging Modality Task:**")
            enhanced_parts.append("- Identify the type of medical imaging")
            enhanced_parts.append("- Common modalities: MRI, CT, X-ray, Ultrasound")
            enhanced_parts.append("- Consider image characteristics")
        
        elif self.task_type == 'lesion_localization':
            enhanced_parts.append("\n**Localization Task:**")
            enhanced_parts.append("- Identify the location of the finding")
            enhanced_parts.append("- Use anatomical landmarks")
            enhanced_parts.append("- Specify laterality if applicable (left/right/bilateral)")
        
        elif self.task_type == 'medical_verification':
            enhanced_parts.append("\n**Medical Yes/No Verification Task:**")
            enhanced_parts.append("- Carefully verify the medical statement")
            enhanced_parts.append("- Answer only 'yes' or 'no'")
            enhanced_parts.append("- Consider all visible findings")
        
        # 医学实体提示
        if self.medical_entities:
            enhanced_parts.append("\n**Relevant Medical Entities:**")
            if self.medical_entities.get('anatomical_structures'):
                enhanced_parts.append(f"- Anatomy: {', '.join(self.medical_entities['anatomical_structures'])}")
            if self.medical_entities.get('abnormalities'):
                enhanced_parts.append(f"- Abnormalities: {', '.join(self.medical_entities['abnormalities'])}")
            if self.medical_entities.get('modalities'):
                enhanced_parts.append(f"- Modalities: {', '.join(self.medical_entities['modalities'])}")
            if self.medical_entities.get('locations'):
                enhanced_parts.append(f"- Locations: {', '.join(self.medical_entities['locations'])}")
        
        # 答案格式指导
        enhanced_parts.append("\n**Answer Format:**")
        if self.is_binary:
            enhanced_parts.append("- Answer ONLY 'yes' or 'no'")
            enhanced_parts.append("- No medical explanation needed")
        else:
            enhanced_parts.append(f"- Expected answer length: {self.answer_length_category}")
            if self.answer_length_category == 'short':
                enhanced_parts.append("- Provide a concise medical term or short answer")
            elif self.answer_length_category == 'medium':
                enhanced_parts.append("- Provide a brief medical answer (2-3 words)")
            else:
                enhanced_parts.append("- Provide a complete medical description")
        
        # 临床相关性提示
        if self.is_clinical:
            enhanced_parts.append("\n**⚠️ Clinical Relevance:**")
            enhanced_parts.append("- This question has clinical significance")
            enhanced_parts.append("- Consider the medical implications")
        
        # 医学复杂度警告
        if self.is_complex or self.medical_complexity.get('has_medical_terms'):
            enhanced_parts.append("\n**⚠️ Medical Complexity Warning:**")
            enhanced_parts.append("- This question involves medical terminology")
            if self.medical_complexity.get('has_laterality'):
                enhanced_parts.append("- Pay attention to laterality (left/right/bilateral)")
            if self.medical_complexity.get('is_differential'):
                enhanced_parts.append("- Consider differential diagnosis")
        
        # 常见医学错误提醒
        enhanced_parts.append("\n**Common Medical Mistakes to Avoid:**")
        if self.task_type == 'abnormality_detection':
            enhanced_parts.append("- Don't miss subtle findings")
            enhanced_parts.append("- Consider the entire visible area")
        elif self.task_type == 'anatomical_recognition':
            enhanced_parts.append("- Use correct anatomical terminology")
            enhanced_parts.append("- Don't confuse similar structures")
        elif self.task_type == 'modality_recognition':
            enhanced_parts.append("- Distinguish between different imaging types")
            enhanced_parts.append("- Consider contrast enhancement")
        elif self.task_type == 'medical_verification':
            enhanced_parts.append("- Carefully read the medical statement")
            enhanced_parts.append("- Look for any contradicting findings")
        
        return "\n".join(enhanced_parts)
    
    def _analyze_medical_features(self) -> Dict[str, bool]:
        """分析医学问题特征"""
        if not self.question:
            return {}
        
        q_lower = self.question.lower()
        
        features = {
            'asks_diagnosis': any(term in q_lower for term in ['diagnosis', 'condition', 'disease', 'syndrome']),
            'asks_anatomy': any(term in q_lower for term in ['organ', 'structure', 'anatomy', 'system']),
            'asks_abnormality': any(term in q_lower for term in ['abnormal', 'lesion', 'mass', 'finding']),
            'asks_modality': any(term in q_lower for term in ['modality', 'scan', 'mri', 'ct', 'x-ray']),
            'asks_location': 'where' in q_lower or 'location' in q_lower,
            'has_medical_terms': any(term in q_lower for term in self.MEDICAL_COMPLEXITY_MARKERS['medical_terms']),
            'has_anatomical_terms': any(term in q_lower for term in self.MEDICAL_COMPLEXITY_MARKERS['anatomical_terms']),
            'has_laterality': any(term in q_lower for term in self.MEDICAL_COMPLEXITY_MARKERS['laterality_terms']),
            'is_differential': any(marker in q_lower for marker in self.MEDICAL_COMPLEXITY_MARKERS['differential_markers']),
            'is_yes_no': q_lower.startswith(('is', 'are', 'does', 'do', 'can', 'has', 'have')),
            'asks_normal': 'normal' in q_lower or 'abnormal' in q_lower,
            'asks_size': any(term in q_lower for term in ['size', 'dimension', 'measure', 'large', 'small'])
        }
        
        return features
    
    def check_success(self, action: Any) -> Tuple[bool, str]:
        """
        检查医学VQA答案
        
        支持yes/no答案和开放式医学答案
        """
        if action is None:
            return False, "No answer provided"
        
        # 规范化用户答案
        user_answer = self._normalize_medical_answer(action)
        
        if user_answer is None or user_answer == "":
            return False, "Please provide a valid medical answer"
        
        # 获取正确答案
        correct_answer = self.expected_answer
        
        # 检查是否匹配
        if user_answer == correct_answer:
            clinical_info = ""
            if self.is_clinical:
                clinical_info = " (Clinically relevant)"
            return True, f"Correct! The answer is '{correct_answer}'{clinical_info}"
        else:
            # 对于开放式医学问题，检查是否是可接受的医学变体
            if not self.is_binary and self._is_acceptable_medical_variant(user_answer, correct_answer):
                return True, f"Acceptable medical answer! (Answer: '{correct_answer}', you said: '{user_answer}')"
            
            # 提供有用的医学反馈
            if self.is_binary:
                explanation = self._get_medical_error_explanation(user_answer, correct_answer)
                return False, f"Incorrect. The answer is '{correct_answer}'. {explanation}"
            else:
                # 对于开放式医学答案，提供更多上下文
                medical_context = ""
                if self.is_clinical:
                    medical_context = " This has clinical significance."
                return False, f"Incorrect. The medical answer is '{correct_answer}' (you said '{user_answer}').{medical_context}"
    
    def _normalize_medical_answer(self, answer: Any) -> Optional[str]:
        """规范化医学答案格式"""
        if answer is None:
            return None
        
        answer_str = str(answer).strip().lower()
        
        # 移除标点符号（但保留医学术语中的连字符）
        answer_str = re.sub(r'[.,!?;:\'"]+', '', answer_str)
        
        # 对于二元答案
        if self.is_binary:
            # 处理常见的yes变体
            if answer_str in ['yes', 'y', 'true', '1', 'correct', 'positive', 'present', 'abnormal']:
                return 'yes'
            # 处理常见的no变体
            elif answer_str in ['no', 'n', 'false', '0', 'incorrect', 'negative', 'absent', 'normal']:
                return 'no'
            # 尝试从句子中提取yes/no
            elif re.search(r'\b(yes|present|positive|abnormal)\b', answer_str) and not re.search(r'\b(no|absent|negative|normal)\b', answer_str):
                return 'yes'
            elif re.search(r'\b(no|absent|negative|normal)\b', answer_str) and not re.search(r'\b(yes|present|positive|abnormal)\b', answer_str):
                return 'no'
            else:
                return None
        else:
            # 对于开放式医学答案，进行基本清理
            # 移除冠词（但保留医学术语中的）
            answer_str = re.sub(r'\b(a|an|the)\s+', '', answer_str).strip()
            # 移除多余空格
            answer_str = ' '.join(answer_str.split())
            return answer_str
    
    def _is_acceptable_medical_variant(self, user_answer: str, correct_answer: str) -> bool:
        """检查是否是可接受的医学答案变体"""
        # 完全匹配
        if user_answer == correct_answer:
            return True
        
        # 医学同义词检查
        medical_synonyms = {
            'mri': ['magnetic resonance imaging', 'mr', 'mr imaging'],
            'ct': ['computed tomography', 'cat scan', 'ct scan'],
            'x-ray': ['xray', 'radiograph', 'plain film'],
            'pneumonia': ['lung infection', 'pulmonary infection'],
            'cardiomegaly': ['enlarged heart', 'cardiac enlargement'],
            'effusion': ['fluid', 'fluid collection'],
            'mass': ['tumor', 'lesion', 'nodule'],
            'hemorrhage': ['bleeding', 'blood'],
            'fracture': ['break', 'broken bone'],
            'normal': ['no abnormality', 'unremarkable', 'within normal limits'],
            'abnormal': ['pathologic', 'pathological', 'diseased']
        }
        
        # 检查同义词
        for key, values in medical_synonyms.items():
            if correct_answer == key and user_answer in values:
                return True
            if user_answer == key and correct_answer in values:
                return True
            # 检查部分匹配
            for value in values:
                if (key in correct_answer and value in user_answer) or (value in correct_answer and key in user_answer):
                    return True
        
        # 侧向性的灵活匹配
        if 'left' in correct_answer and 'left' in user_answer:
            return True
        if 'right' in correct_answer and 'right' in user_answer:
            return True
        if 'bilateral' in correct_answer and ('bilateral' in user_answer or ('left' in user_answer and 'right' in user_answer)):
            return True
        
        return False
    
    def _get_medical_error_explanation(self, user_answer: str, correct_answer: str) -> str:
        """生成医学错误解释（主要用于yes/no问题）"""
        explanations = []
        
        if self.task_type == 'medical_verification':
            if correct_answer == 'yes':
                explanations.append("The medical finding IS present in the image.")
            else:
                explanations.append("The medical finding is NOT present in the image.")
        
        if self.task_type == 'abnormality_detection':
            if correct_answer == 'yes':
                explanations.append("There ARE abnormalities visible.")
            else:
                explanations.append("There are NO abnormalities visible.")
        
        if self.medical_complexity.get('has_negation'):
            explanations.append("Note the negation in the medical question.")
        
        if self.is_clinical:
            explanations.append("This has clinical significance.")
        
        return " ".join(explanations) if explanations else ""
    
    def validate(
        self,
        chat_history: List[Dict],
        observation: Any,
        full_history: Optional[List[Any]] = None
    ) -> Tuple[float, bool, str, Dict[str, Any]]:
        """
        验证医学VQA任务执行情况
        """
        # 调用父类验证
        reward, done, message, info = super().validate(
            chat_history, observation, full_history
        )
        
        # 添加医学VQA特定的信息
        info["task_type"] = self.task_type
        info["question_type"] = self.question_type
        info["answer_type"] = self.answer_type
        info["expected_answer"] = self.expected_answer
        info["is_binary"] = self.is_binary
        info["is_clinical"] = self.is_clinical
        info["is_complex"] = self.is_complex
        info["answer_length"] = self.answer_length_category
        info["medical_entities"] = self.medical_entities
        info["medical_complexity"] = self.medical_complexity
        info["medical_domain"] = self.metadata.get('medical_domain', 'radiology')
        info["difficulty"] = self._assess_medical_difficulty()
        
        # 分析医学错误类型（如果答案错误）
        if not info.get("success", False) and info.get("answer_provided"):
            error_analysis = self._analyze_medical_error(info["answer_provided"])
            info["error_analysis"] = error_analysis
        
        return reward, done, message, info
    
    def _analyze_medical_error(self, user_answer: Any) -> Dict[str, Any]:
        """分析医学错误类型"""
        error_info = {
            "error_type": "unknown",
            "user_answer": str(user_answer),
            "correct_answer": self.expected_answer,
            "normalized_user_answer": None,
            "is_medical_term": False
        }
        
        # 规范化用户答案
        normalized = self._normalize_medical_answer(user_answer)
        error_info["normalized_user_answer"] = normalized
        
        if normalized is None or normalized == "":
            error_info["error_type"] = "invalid_medical_format"
        else:
            # 检查是否是医学术语
            medical_terms = (
                self.ANSWER_CATEGORIES['modalities'] + 
                self.ANSWER_CATEGORIES['anatomical_structures'] +
                self.ANSWER_CATEGORIES['abnormalities'] +
                self.ANSWER_CATEGORIES['conditions']
            )
            if any(term in normalized for term in medical_terms):
                error_info["is_medical_term"] = True
            
            # 分析错误类型
            if self.is_binary:
                error_info["error_type"] = "medical_binary_confusion"
            elif self.task_type == 'modality_recognition':
                if normalized in self.ANSWER_CATEGORIES['modalities']:
                    error_info["error_type"] = "wrong_modality"
                else:
                    error_info["error_type"] = "invalid_modality"
            elif self.task_type == 'anatomical_recognition':
                if any(anat in normalized for anat in self.ANSWER_CATEGORIES['anatomical_structures']):
                    error_info["error_type"] = "wrong_anatomy"
                else:
                    error_info["error_type"] = "anatomy_misidentification"
            elif self.task_type == 'abnormality_detection':
                if any(abnorm in normalized for abnorm in self.ANSWER_CATEGORIES['abnormalities']):
                    error_info["error_type"] = "wrong_abnormality_type"
                else:
                    error_info["error_type"] = "abnormality_misinterpretation"
            elif self.task_type == 'medical_diagnosis':
                error_info["error_type"] = "incorrect_diagnosis"
            else:
                error_info["error_type"] = "medical_content_mismatch"
        
        return error_info
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取医学VQA特定的指标"""
        metrics = super().get_metrics()
        
        metrics.update({
            "task_type": self.task_type,
            "question_type": self.question_type,
            "answer_type": self.answer_type,
            "expected_answer": self.expected_answer,
            "is_binary": self.is_binary,
            "is_clinical": self.is_clinical,
            "is_complex": self.is_complex,
            "answer_length": self.answer_length_category,
            "medical_domain": self.metadata.get('medical_domain', 'radiology'),
            "difficulty": self._assess_medical_difficulty(),
            "medical_features": self._analyze_medical_features(),
            "medical_entities": self.medical_entities,
            "medical_complexity": self.medical_complexity
        })
        
        return metrics
    
    def _assess_medical_difficulty(self) -> str:
        """评估医学任务难度"""
        difficulty_score = 0
        
        # 基于临床相关性
        if self.is_clinical:
            difficulty_score += 2
        
        # 基于医学复杂度
        if self.is_complex:
            difficulty_score += 2
        
        if self.medical_complexity.get('has_medical_terms'):
            difficulty_score += 1
        
        if self.medical_complexity.get('is_differential'):
            difficulty_score += 2
        
        # 基于答案长度
        if self.answer_length_category == 'long':
            difficulty_score += 2
        elif self.answer_length_category == 'medium':
            difficulty_score += 1
        
        # 基于任务类型
        difficult_tasks = ['medical_diagnosis', 'medical_description', 'measurement_estimation']
        if self.task_type in difficult_tasks:
            difficulty_score += 2
        
        # 基于是否开放式
        if not self.is_binary:
            difficulty_score += 1
        
        # 返回难度等级
        if difficulty_score >= 6:
            return "hard"
        elif difficulty_score >= 3:
            return "medium"
        else:
            return "easy"
    
    def get_observation(self) -> Dict[str, Any]:
        """获取任务观察"""
        obs = super().get_observation()
        
        # 添加医学VQA特定信息
        obs["scene_type"] = "Medical radiology image"
        obs["expected_content"] = "anatomical structures, medical findings, abnormalities"
        obs["task_focus"] = "medical image understanding and clinical question answering"
        obs["answer_type"] = self.answer_type
        obs["answer_format"] = "binary (yes/no)" if self.is_binary else f"medical term ({self.answer_length_category} answer)"
        obs["clinical_relevance"] = "yes" if self.is_clinical else "no"
        obs["medical_domain"] = self.metadata.get('medical_domain', 'radiology')
        
        return obs
    
    def get_task_description(self) -> str:
        """获取任务描述"""
        desc_parts = [
            f"This is a Medical Visual Question Answering (VQA-RAD) task on a radiology image.",
            f"Medical domain: {self.metadata.get('medical_domain', 'radiology')}",
            f"Task type: {self.TASK_TYPES.get(self.task_type, self.task_type)}",
            f"Question type: {self.QUESTION_TYPES.get(self.question_type, self.question_type)}"
        ]
        
        if self.is_clinical:
            desc_parts.append("⚠️ This question has clinical relevance.")
        
        if self.is_complex:
            desc_parts.append("⚠️ This is a complex medical question requiring careful analysis.")
        
        if self.is_binary:
            desc_parts.append(f"Expected answer: '{self.expected_answer}' (yes/no)")
        else:
            desc_parts.append(f"Expected answer: '{self.expected_answer}' ({self.answer_length_category} medical answer)")
        
        # 添加医学实体信息
        if self.medical_entities:
            entities_info = []
            if self.medical_entities.get('anatomical_structures'):
                entities_info.append(f"Anatomy: {', '.join(self.medical_entities['anatomical_structures'])}")
            if self.medical_entities.get('abnormalities'):
                entities_info.append(f"Abnormalities: {', '.join(self.medical_entities['abnormalities'])}")
            if self.medical_entities.get('modalities'):
                entities_info.append(f"Modality: {', '.join(self.medical_entities['modalities'])}")
            if entities_info:
                desc_parts.append(f"Medical entities: {'; '.join(entities_info)}")
        
        desc_parts.append(f"Difficulty: {self._assess_medical_difficulty()}")
        
        return "\n".join(desc_parts)