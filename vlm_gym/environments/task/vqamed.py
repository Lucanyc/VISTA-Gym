"""
VQA-Med Task implementation for VLM Gym
Handles Medical Visual Question Answering tasks focusing on imaging modalities and technical aspects
"""

from typing import Tuple, Dict, Any, List, Optional
import re
import json
from pathlib import Path
from collections import Counter

from .vision_qa_task import VisionQATask


class VqaMedTask(VisionQATask):
    """
    VQA-Med (Visual Question Answering - Medical) 特定任务
    
    专门处理医学影像的视觉问答任务，特别关注：
    - 成像模态识别（what kind of image, what modality）
    - 成像技术问题（contrast/noncontrast, weighted sequences）
    - 解剖系统识别（anatomical systems）
    - 成像平面识别（axial, sagittal, coronal）
    - 医学验证（is this a T1 weighted image?）
    - 技术参数问题（acquisition parameters）
    
    支持二元判断（yes/no）和开放式医学技术答案
    """
    
    # 医学任务类型（针对VQA-Med特点调整）
    TASK_TYPES = {
        'modality_recognition': '成像模态识别',
        'medical_verification': '医学验证（Yes/No）',
        'abnormality_detection': '异常检测',
        'medical_vqa': '通用医学视觉问答',
        'anatomical_localization': '解剖定位',
        'medical_counting': '医学计数',
        'medical_description': '医学描述',
        'measurement_estimation': '测量估计'
    }
    
    # 医学问题类型（针对VQA-Med特点）
    QUESTION_TYPES = {
        'modality': '模态类（what kind of image, modality）',
        'yes_no': 'Yes/No判断类',
        'abnormality': '异常类（abnormal findings）',
        'location': '位置类（where, which region）',
        'counting': '计数类（how many）',
        'description': '描述类（describe）',
        'measurement': '测量类（size）',
        'general': '通用类'
    }
    
    # VQA-Med特定的答案类别
    ANSWER_CATEGORIES = {
        'yes_no': ['yes', 'no'],
        'modalities': ['ct', 'mri', 'xr - plain film', 'us - ultrasound', 'cta - ct angiography', 
                      'mra - mr angiography', 'pet', 'spect', 'mammography', 'fluoroscopy'],
        'imaging_planes': ['axial', 'sagittal', 'coronal', 'transverse', 'oblique'],
        'anatomical_systems': ['skull and contents', 'musculoskeletal', 'gastrointestinal', 
                              'spine and contents', 'cardiovascular', 'respiratory', 
                              'genitourinary', 'breast', 'head and neck'],
        'contrast_status': ['contrast', 'noncontrast', 'post-contrast', 'pre-contrast'],
        'mri_sequences': ['t1', 't2', 'flair', 'dwi', 'adc', 'stir', 'gre', 'swi'],
        'technical_terms': ['weighted', 'sequence', 'phase', 'acquisition', 'reconstruction']
    }
    
    # 医学技术复杂度标记
    MEDICAL_COMPLEXITY_MARKERS = {
        'modality_terms': ['modality', 'image', 'scan', 'imaging', 'taken with', 'type of'],
        'technical_terms': ['weighted', 'contrast', 'sequence', 'phase', 'acquisition'],
        'mri_terms': ['t1', 't2', 'flair', 'dwi', 'weighted'],
        'contrast_terms': ['contrast', 'noncontrast', 'enhanced', 'enhancement']
    }
    
    def __init__(self, task_id: str, adapter: Any):
        """
        初始化VQA-Med任务
        
        Args:
            task_id: 任务ID
            adapter: VQA-Med数据适配器
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
        self.is_modality_question = False
        self.is_contrast_question = False
        self.is_weighted_question = False
        self.answer_category = None  # 'modality', 'anatomical', 'plane', etc.
        
        # 获取任务数据
        task_data = adapter.get_task_data(task_id)
        self.dataset_name = task_data.get('dataset', 'vqa_med')
        self.task_type = task_data.get('task', 'medical_vqa')
        self.choices = task_data.get('choices', None)
    
    @classmethod
    def get_task_id(cls) -> str:
        """获取任务类型ID"""
        return "vlm-gym.vqa-med"
    
    def setup(self) -> Tuple[str, Dict[str, Any]]:
        """设置VQA-Med特定的任务"""
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
        
        # 判断答案类别
        self._categorize_answer()
        
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
        self.is_modality_question = medical_features.get('asks_modality', False)
        self.is_contrast_question = medical_features.get('asks_contrast', False)
        self.is_weighted_question = medical_features.get('asks_weighted', False)
        
        # 添加医学VQA特定的信息
        task_info["task_type"] = self.task_type
        task_info["question_type"] = self.question_type
        task_info["answer_type"] = self.answer_type
        task_info["expected_answer"] = self.expected_answer
        task_info["answer_category"] = self.answer_category
        task_info["is_binary"] = self.is_binary
        task_info["is_clinical"] = self.is_clinical
        task_info["is_complex"] = self.is_complex
        task_info["answer_length"] = self.answer_length_category
        task_info["medical_features"] = medical_features
        task_info["medical_entities"] = self.medical_entities
        task_info["medical_complexity"] = self.medical_complexity
        task_info["dataset"] = "vqa_med"
        task_info["medical_domain"] = self.metadata.get('medical_domain', 'radiology')
        task_info["difficulty"] = self._assess_medical_difficulty()
        
        # 修改任务目标以包含医学VQA特定指导
        enhanced_goal = self._enhance_medical_task_goal(task_goal, medical_features)
        
        return enhanced_goal, task_info
    
    def _categorize_answer(self):
        """分类答案类型"""
        if not self.expected_answer:
            self.answer_category = 'unknown'
            return
        
        answer_lower = self.expected_answer.lower()
        
        # 检查是否是模态答案
        if any(modality in answer_lower for modality in ['ct', 'mri', 'xr', 'ultrasound', 'angiography']):
            self.answer_category = 'modality'
        # 检查是否是解剖系统答案
        elif any(system in answer_lower for system in self.ANSWER_CATEGORIES['anatomical_systems']):
            self.answer_category = 'anatomical'
        # 检查是否是成像平面答案
        elif any(plane in answer_lower for plane in self.ANSWER_CATEGORIES['imaging_planes']):
            self.answer_category = 'plane'
        # 检查是否是对比状态答案
        elif any(contrast in answer_lower for contrast in self.ANSWER_CATEGORIES['contrast_status']):
            self.answer_category = 'contrast'
        # 检查是否是MRI序列答案
        elif any(seq in answer_lower for seq in self.ANSWER_CATEGORIES['mri_sequences']):
            self.answer_category = 'mri_sequence'
        # 检查是否是yes/no答案
        elif answer_lower in ['yes', 'no']:
            self.answer_category = 'yes_no'
        else:
            self.answer_category = 'other'
    
    def _enhance_medical_task_goal(self, base_goal: str, features: Dict[str, bool]) -> str:
        """增强任务目标描述，添加医学VQA特定的指导"""
        enhanced_parts = [base_goal]
        
        # 添加医学影像理解通用指导
        enhanced_parts.append("\n**Medical Image Understanding:**")
        enhanced_parts.append("1. Carefully observe the medical image")
        enhanced_parts.append("2. Identify the imaging modality and technical characteristics")
        enhanced_parts.append("3. Consider acquisition parameters if relevant")
        enhanced_parts.append("4. Apply medical imaging knowledge to answer the question")
        
        # 根据任务类型添加特定指导
        if self.task_type == 'modality_recognition' or self.is_modality_question:
            enhanced_parts.append("\n**Modality Recognition Task:**")
            enhanced_parts.append("- Identify the type of medical imaging")
            enhanced_parts.append("- Common modalities: CT, MRI, X-ray (XR), Ultrasound (US)")
            enhanced_parts.append("- Consider contrast enhancement (CTA, MRA)")
            enhanced_parts.append("- Use standard abbreviations when appropriate")
        
        elif self.task_type == 'medical_verification':
            enhanced_parts.append("\n**Medical Verification Task:**")
            enhanced_parts.append("- Carefully verify the medical/technical statement")
            enhanced_parts.append("- Answer only 'yes' or 'no'")
            enhanced_parts.append("- Consider imaging characteristics")
        
        # 特定问题类型指导
        if self.is_contrast_question:
            enhanced_parts.append("\n**Contrast Assessment:**")
            enhanced_parts.append("- Determine if contrast agent was used")
            enhanced_parts.append("- Look for enhancement patterns")
            enhanced_parts.append("- Answer: 'contrast' or 'noncontrast'")
        
        if self.is_weighted_question:
            enhanced_parts.append("\n**MRI Sequence Identification:**")
            enhanced_parts.append("- Identify MRI weighting (T1, T2, FLAIR, etc.)")
            enhanced_parts.append("- Consider tissue signal characteristics")
            enhanced_parts.append("- Look for sequence-specific features")
        
        # 医学实体提示
        if self.medical_entities:
            enhanced_parts.append("\n**Relevant Medical Entities:**")
            if self.medical_entities.get('modalities'):
                enhanced_parts.append(f"- Modalities: {', '.join(self.medical_entities['modalities'])}")
            if self.medical_entities.get('imaging_characteristics'):
                enhanced_parts.append(f"- Imaging features: {', '.join(self.medical_entities['imaging_characteristics'])}")
            if self.medical_entities.get('technical_terms'):
                enhanced_parts.append(f"- Technical terms: {', '.join(self.medical_entities['technical_terms'])}")
        
        # 答案格式指导
        enhanced_parts.append("\n**Answer Format:**")
        if self.is_binary:
            enhanced_parts.append("- Answer ONLY 'yes' or 'no'")
            enhanced_parts.append("- No explanation needed")
        else:
            enhanced_parts.append(f"- Expected answer type: {self.answer_category}")
            enhanced_parts.append(f"- Expected answer length: {self.answer_length_category}")
            if self.answer_category == 'modality':
                enhanced_parts.append("- Use standard modality abbreviations (e.g., 'xr - plain film', 'ct', 'mri')")
            elif self.answer_category == 'anatomical':
                enhanced_parts.append("- Provide anatomical system name")
            elif self.answer_category == 'plane':
                enhanced_parts.append("- Specify imaging plane (axial, sagittal, coronal)")
        
        # 临床相关性提示
        if self.is_clinical:
            enhanced_parts.append("\n**⚠️ Clinical Relevance:**")
            enhanced_parts.append("- This question has clinical/technical significance")
        
        # 常见错误提醒
        enhanced_parts.append("\n**Common Mistakes to Avoid:**")
        if self.task_type == 'modality_recognition':
            enhanced_parts.append("- Don't confuse similar modalities (CT vs MRI)")
            enhanced_parts.append("- Include contrast specification when relevant (e.g., 'cta - ct angiography')")
        elif self.is_contrast_question:
            enhanced_parts.append("- Distinguish between pre- and post-contrast images")
            enhanced_parts.append("- Look for enhancement patterns")
        elif self.is_weighted_question:
            enhanced_parts.append("- Don't confuse T1 and T2 weighting")
            enhanced_parts.append("- Consider tissue signal intensity")
        
        return "\n".join(enhanced_parts)
    
    def _analyze_medical_features(self) -> Dict[str, bool]:
        """分析医学问题特征"""
        if not self.question:
            return {}
        
        q_lower = self.question.lower()
        
        features = {
            'asks_modality': any(term in q_lower for term in ['modality', 'kind of image', 'type of image', 
                                                               'what is this image', 'taken with', 'imaging modality']),
            'asks_contrast': any(term in q_lower for term in ['contrast', 'noncontrast', 'enhanced']),
            'asks_weighted': any(term in q_lower for term in ['weighted', 't1', 't2', 'flair', 'sequence']),
            'asks_plane': any(term in q_lower for term in ['plane', 'view', 'axial', 'sagittal', 'coronal']),
            'is_yes_no': q_lower.startswith(('is', 'are', 'does', 'do', 'can', 'has', 'have')),
            'has_technical_terms': any(term in q_lower for term in self.MEDICAL_COMPLEXITY_MARKERS['technical_terms']),
            'has_modality_terms': any(term in q_lower for term in self.MEDICAL_COMPLEXITY_MARKERS['modality_terms']),
            'has_mri_terms': any(term in q_lower for term in self.MEDICAL_COMPLEXITY_MARKERS['mri_terms']),
            'has_contrast_terms': any(term in q_lower for term in self.MEDICAL_COMPLEXITY_MARKERS['contrast_terms'])
        }
        
        return features
    
    def check_success(self, action: Any) -> Tuple[bool, str]:
        """
        检查医学VQA答案
        
        支持yes/no答案和开放式医学技术答案
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
                category_hint = f" ({self.answer_category} answer)" if self.answer_category else ""
                return False, f"Incorrect. The answer is '{correct_answer}'{category_hint} (you said '{user_answer}')."
    
    def _normalize_medical_answer(self, answer: Any) -> Optional[str]:
        """规范化医学答案格式"""
        if answer is None:
            return None
        
        answer_str = str(answer).strip().lower()
        
        # 移除标点符号（但保留医学术语中的连字符和空格）
        answer_str = re.sub(r'[.,!?;:\'"]+', '', answer_str)
        
        # 对于二元答案
        if self.is_binary:
            # 处理常见的yes变体
            if answer_str in ['yes', 'y', 'true', '1', 'correct', 'positive', 'present']:
                return 'yes'
            # 处理常见的no变体
            elif answer_str in ['no', 'n', 'false', '0', 'incorrect', 'negative', 'absent']:
                return 'no'
            # 尝试从句子中提取yes/no
            elif re.search(r'\b(yes|present|positive)\b', answer_str) and not re.search(r'\b(no|absent|negative)\b', answer_str):
                return 'yes'
            elif re.search(r'\b(no|absent|negative)\b', answer_str) and not re.search(r'\b(yes|present|positive)\b', answer_str):
                return 'no'
            else:
                return None
        else:
            # 对于开放式医学答案
            # 标准化模态名称
            modality_mappings = {
                'x-ray': 'xr - plain film',
                'xray': 'xr - plain film',
                'plain film': 'xr - plain film',
                'radiograph': 'xr - plain film',
                'ultrasound': 'us - ultrasound',
                'us': 'us - ultrasound',
                'ct angiography': 'cta - ct angiography',
                'cta': 'cta - ct angiography',
                'mr angiography': 'mra - mr angiography',
                'mra': 'mra - mr angiography'
            }
            
            # 检查是否需要映射
            for key, value in modality_mappings.items():
                if key in answer_str:
                    answer_str = answer_str.replace(key, value)
            
            # 移除冠词
            answer_str = re.sub(r'\b(a|an|the)\s+', '', answer_str).strip()
            # 移除多余空格
            answer_str = ' '.join(answer_str.split())
            return answer_str
    
    def _is_acceptable_medical_variant(self, user_answer: str, correct_answer: str) -> bool:
        """检查是否是可接受的医学答案变体"""
        # 完全匹配
        if user_answer == correct_answer:
            return True
        
        # VQA-Med特定的同义词
        medical_synonyms = {
            'xr - plain film': ['x-ray', 'xray', 'radiograph', 'plain film', 'radiography'],
            'us - ultrasound': ['ultrasound', 'us', 'sonography', 'ultrasonography'],
            'cta - ct angiography': ['ct angiography', 'cta', 'computed tomography angiography'],
            'mra - mr angiography': ['mr angiography', 'mra', 'magnetic resonance angiography'],
            'ct': ['computed tomography', 'cat scan', 'ct scan'],
            'mri': ['magnetic resonance imaging', 'mr', 'mr imaging'],
            'contrast': ['with contrast', 'post-contrast', 'enhanced'],
            'noncontrast': ['without contrast', 'non-contrast', 'unenhanced', 'pre-contrast']
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
        
        # 检查解剖系统的部分匹配
        for system in self.ANSWER_CATEGORIES['anatomical_systems']:
            if system in correct_answer and system in user_answer:
                return True
        
        # 检查成像平面的匹配
        for plane in self.ANSWER_CATEGORIES['imaging_planes']:
            if plane in correct_answer and plane in user_answer:
                return True
        
        return False
    
    def _get_medical_error_explanation(self, user_answer: str, correct_answer: str) -> str:
        """生成医学错误解释（主要用于yes/no问题）"""
        explanations = []
        
        if self.is_modality_question:
            if correct_answer == 'yes':
                explanations.append("This IS the specified imaging modality.")
            else:
                explanations.append("This is NOT the specified imaging modality.")
        
        if self.is_contrast_question:
            if correct_answer == 'yes':
                explanations.append("Contrast WAS used in this image.")
            else:
                explanations.append("Contrast was NOT used in this image.")
        
        if self.is_weighted_question:
            if correct_answer == 'yes':
                explanations.append("This IS the specified MRI sequence.")
            else:
                explanations.append("This is NOT the specified MRI sequence.")
        
        if self.medical_complexity.get('has_technical_terms'):
            explanations.append("Pay attention to technical imaging details.")
        
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
        info["answer_category"] = self.answer_category
        info["expected_answer"] = self.expected_answer
        info["is_binary"] = self.is_binary
        info["is_clinical"] = self.is_clinical
        info["is_complex"] = self.is_complex
        info["answer_length"] = self.answer_length_category
        info["medical_entities"] = self.medical_entities
        info["medical_complexity"] = self.medical_complexity
        info["medical_domain"] = self.metadata.get('medical_domain', 'radiology')
        info["difficulty"] = self._assess_medical_difficulty()
        info["is_modality_question"] = self.is_modality_question
        info["is_contrast_question"] = self.is_contrast_question
        info["is_weighted_question"] = self.is_weighted_question
        
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
            "answer_category": self.answer_category
        }
        
        # 规范化用户答案
        normalized = self._normalize_medical_answer(user_answer)
        error_info["normalized_user_answer"] = normalized
        
        if normalized is None or normalized == "":
            error_info["error_type"] = "invalid_format"
        else:
            # 分析错误类型
            if self.is_binary:
                error_info["error_type"] = "binary_confusion"
            elif self.answer_category == 'modality':
                if any(mod in normalized for mod in ['ct', 'mri', 'xr', 'us']):
                    error_info["error_type"] = "wrong_modality"
                else:
                    error_info["error_type"] = "invalid_modality"
            elif self.answer_category == 'anatomical':
                if any(sys in normalized for sys in self.ANSWER_CATEGORIES['anatomical_systems']):
                    error_info["error_type"] = "wrong_anatomical_system"
                else:
                    error_info["error_type"] = "invalid_anatomical_answer"
            elif self.answer_category == 'plane':
                if any(plane in normalized for plane in self.ANSWER_CATEGORIES['imaging_planes']):
                    error_info["error_type"] = "wrong_imaging_plane"
                else:
                    error_info["error_type"] = "invalid_plane_answer"
            elif self.answer_category == 'contrast':
                error_info["error_type"] = "contrast_status_error"
            else:
                error_info["error_type"] = "content_mismatch"
        
        return error_info
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取医学VQA特定的指标"""
        metrics = super().get_metrics()
        
        metrics.update({
            "task_type": self.task_type,
            "question_type": self.question_type,
            "answer_type": self.answer_type,
            "answer_category": self.answer_category,
            "expected_answer": self.expected_answer,
            "is_binary": self.is_binary,
            "is_clinical": self.is_clinical,
            "is_complex": self.is_complex,
            "answer_length": self.answer_length_category,
            "medical_domain": self.metadata.get('medical_domain', 'radiology'),
            "difficulty": self._assess_medical_difficulty(),
            "medical_features": self._analyze_medical_features(),
            "medical_entities": self.medical_entities,
            "medical_complexity": self.medical_complexity,
            "is_modality_question": self.is_modality_question,
            "is_contrast_question": self.is_contrast_question,
            "is_weighted_question": self.is_weighted_question
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
        
        if self.medical_complexity.get('is_technical'):
            difficulty_score += 1
        
        if self.medical_complexity.get('has_mri_terms'):
            difficulty_score += 1
        
        # 基于答案类别
        if self.answer_category in ['mri_sequence', 'technical']:
            difficulty_score += 2
        elif self.answer_category in ['contrast', 'plane']:
            difficulty_score += 1
        
        # 基于答案长度
        if self.answer_length_category == 'long':
            difficulty_score += 1
        
        # 基于任务类型
        if self.task_type in ['medical_description', 'measurement_estimation']:
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
        obs["scene_type"] = "Medical imaging scan"
        obs["expected_content"] = "imaging modality characteristics, technical parameters, anatomical structures"
        obs["task_focus"] = "medical imaging modality recognition and technical assessment"
        obs["answer_type"] = self.answer_type
        obs["answer_category"] = self.answer_category
        obs["answer_format"] = "binary (yes/no)" if self.is_binary else f"medical term ({self.answer_category})"
        obs["clinical_relevance"] = "yes" if self.is_clinical else "no"
        obs["medical_domain"] = self.metadata.get('medical_domain', 'radiology')
        
        return obs
    
    def get_task_description(self) -> str:
        """获取任务描述"""
        desc_parts = [
            f"This is a Medical Visual Question Answering (VQA-Med) task.",
            f"Medical domain: {self.metadata.get('medical_domain', 'radiology')}",
            f"Task type: {self.TASK_TYPES.get(self.task_type, self.task_type)}",
            f"Question type: {self.QUESTION_TYPES.get(self.question_type, self.question_type)}"
        ]
        
        if self.is_modality_question:
            desc_parts.append("📷 This is a modality recognition question.")
        
        if self.is_contrast_question:
            desc_parts.append("💉 This question asks about contrast enhancement.")
        
        if self.is_weighted_question:
            desc_parts.append("🧲 This question asks about MRI sequence/weighting.")
        
        if self.is_clinical:
            desc_parts.append("⚠️ This question has clinical/technical relevance.")
        
        if self.is_complex:
            desc_parts.append("⚠️ This is a complex technical question.")
        
        if self.is_binary:
            desc_parts.append(f"Expected answer: '{self.expected_answer}' (yes/no)")
        else:
            desc_parts.append(f"Expected answer: '{self.expected_answer}' ({self.answer_category})")
        
        # 添加医学实体信息
        if self.medical_entities:
            entities_info = []
            if self.medical_entities.get('modalities'):
                entities_info.append(f"Modalities: {', '.join(self.medical_entities['modalities'])}")
            if self.medical_entities.get('imaging_characteristics'):
                entities_info.append(f"Imaging: {', '.join(self.medical_entities['imaging_characteristics'])}")
            if self.medical_entities.get('technical_terms'):
                entities_info.append(f"Technical: {', '.join(self.medical_entities['technical_terms'])}")
            if entities_info:
                desc_parts.append(f"Medical entities: {'; '.join(entities_info)}")
        
        desc_parts.append(f"Difficulty: {self._assess_medical_difficulty()}")
        
        return "\n".join(desc_parts)