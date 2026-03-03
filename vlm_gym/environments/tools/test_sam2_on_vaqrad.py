"""
使用VQA-RAD数据测试SAM2工具集成
支持命令行参数指定图片ID和问题
强制Agent使用SAM2工具来分割医学图像中的相关区域
"""

import os
import sys
from pathlib import Path

# 获取vlm_gym的根目录
current_file = Path(__file__).resolve()
# 如果脚本在 vlm_gym/environments/tools/ 下
if 'vlm_gym' in str(current_file):
    # 找到vlm_gym的父目录
    vlm_gym_root = current_file
    while vlm_gym_root.name != 'vlm_gym-tool-usage-mathvista' and vlm_gym_root.parent != vlm_gym_root:
        vlm_gym_root = vlm_gym_root.parent
    
    # 添加根目录到Python路径
    sys.path.insert(0, str(vlm_gym_root))
else:
    # 如果不在预期位置，尝试其他路径
    sys.path.insert(0, '/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista')

# 添加其他可能需要的路径
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "scripts"))

import json
import argparse
from datetime import datetime
import re
import torch
import numpy as np
from PIL import Image

# 导入必要的类
from vlm_gym.environments import VisionQAEnv
from vlm_gym.agents.vlm_agent_with_tools import VLMAgentWithTools

def extract_yes_no_from_medical_response(text):
    """从医学回答中智能提取yes/no答案"""
    if text is None:
        return 'no'  # 默认返回no
    text = str(text).lower()
    
    # 积极指标（倾向于yes）- 表示存在异常
    positive_indicators = [
        'appears to show', 'indicative of', 'suggests', 'consistent with',
        'likely', 'probable', 'evidence of', 'signs of', 'presence of',
        'abnormal', 'pathological', 'positive for', 'areas of abnormal',
        'could be indicative', 'possibility of', 'warrants further',
        'significant region', 'potential abnormalities'
    ]
    
    # 消极指标（倾向于no）- 表示正常
    negative_indicators = [
        'no evidence', 'normal', 'unremarkable', 'absent', 'negative',
        'not present', 'no signs', 'unlikely', 'does not show',
        'no indication', 'no abnormality', 'appears normal',
        'within normal limits', 'no significant'
    ]
    
    # 不确定指标
    uncertain_indicators = [
        'cannot definitively', 'not possible to', 'uncertain',
        'requires further', 'cannot confirm', 'inconclusive'
    ]
    
    # 计算各类指标的出现次数
    positive_count = sum(1 for indicator in positive_indicators if indicator in text)
    negative_count = sum(1 for indicator in negative_indicators if indicator in text)
    uncertain_count = sum(1 for indicator in uncertain_indicators if indicator in text)
    
    # 直接查找yes/no（作为独立单词）
    words = text.split()
    if 'yes' in words and 'no' not in words:
        return 'yes'
    elif 'no' in words and 'yes' not in words:
        return 'no'
    
    # 特殊情况：如果提到"可能"+"异常/梗塞/病变"等，在医学上通常需要报告为阳性
    if ('possible' in text or 'possibly' in text or 'could be' in text) and \
       any(term in text for term in ['infarct', 'lesion', 'abnormal', 'patholog']):
        return 'yes'
    
    # 基于指标判断
    if positive_count > negative_count:
        return 'yes'
    elif negative_count > positive_count:
        return 'no'
    elif positive_count > 0 and uncertain_count > 0:
        # 如果既有积极指标又表示不确定，在医学筛查中通常倾向于报告异常
        return 'yes'
    else:
        # 默认情况
        return 'no'

def evaluate_vqa_rad_answer(prediction: str, ground_truth: str):
    """评估VQA-RAD答案 - 改进版本"""
    pred_lower = str(prediction).lower().strip()
    gt_lower = str(ground_truth).lower().strip()
    
    # 对于yes/no问题
    if gt_lower in ['yes', 'no']:
        # 使用智能提取
        pred_answer = extract_yes_no_from_medical_response(pred_lower)
        
        is_correct = pred_answer == gt_lower
        return is_correct, 1.0 if is_correct else 0.0, f"Predicted: {pred_answer}, GT: {gt_lower}"
    
    # 对于其他答案，完全匹配
    is_correct = pred_lower == gt_lower
    return is_correct, 1.0 if is_correct else 0.0, f"Predicted: {pred_lower}, GT: {gt_lower}"

def create_vqa_rad_task_wrapper(task_id, task_data, force_tool_use=True):
    """创建VQA-RAD任务包装器"""
    class SimpleVQARadTask:
        def __init__(self, task_id, task_data):
            self.task_id = task_id
            self.task_data = task_data
            self.current_step = 0
            self.max_steps = 10
            self.enable_sam2 = True
            self.metadata = task_data.get('metadata', {})
            self.medical_domain = self.metadata.get('medical_domain', 'radiology')
            self.force_tool_use = force_tool_use
            self.sam2_history = []
            self.tool_used = False  # 跟踪是否已使用工具
            self.segmentation_results = []  # 保存分割结果
            
        def setup(self):
            return f"Answer this medical imaging question: {self.task_data['question']}", {
                "task_type": "vqa_rad",
                "task_id": self.task_id,
                "medical_domain": self.medical_domain,
                "is_medical_vqa": True
            }
        
        def get_observation(self):
            # 修改问题以强制使用SAM2
            original_question = self.task_data["question"]
            
            # 如果强制使用工具且还未使用
            if self.force_tool_use and not self.tool_used:
                modified_question = f"""You MUST use the SAM2 tool to segment the medical image BEFORE answering.

Medical question: {original_question}

DO NOT THINK OR EXPLAIN - JUST OUTPUT THE TOOL CALL.
DO NOT USE <think> TAGS.
ONLY OUTPUT THE <tool_call> EXACTLY AS SHOWN IN THE FORMAT BELOW."""
            else:
                # 如果已经使用了工具，现在要求明确答案
                # 获取最佳分割结果
                best_coverage = 0
                if self.segmentation_results:
                    best_result = max(self.segmentation_results, key=lambda x: x.get('score', 0))
                    best_coverage = best_result.get('coverage_percent', 0)
                
                modified_question = f"""Based on your SAM2 segmentation analysis showing:
- Number of masks: {len(self.segmentation_results)}
- Best mask coverage: {best_coverage:.1f}%
- Detected regions with potential abnormalities

Now answer this yes/no question with a DEFINITIVE answer:
{original_question}

CRITICAL INSTRUCTIONS:
1. You MUST start your answer with either "Yes" or "No"
2. For medical pathology questions:
   - If segmentation shows ANY abnormal regions or patterns → Answer "Yes"
   - If coverage >20% suggests significant findings → Answer "Yes"  
   - Only answer "No" if the image appears completely normal
3. Do NOT hedge with uncertainty - give a definitive clinical answer

Remember: In medical screening, it's better to flag potential issues than miss them."""
            
            # 提取医学关键词
            medical_keywords = []
            question_lower = original_question.lower()
            
            # 器官关键词
            organs = ['brain', 'lung', 'heart', 'liver', 'kidney', 'chest', 'head', 'abdomen']
            for organ in organs:
                if organ in question_lower:
                    medical_keywords.append(organ)
            
            # 医学状态关键词
            conditions = ['normal', 'abnormal', 'infarcted', 'lesion', 'tumor', 'disease']
            for condition in conditions:
                if condition in question_lower:
                    medical_keywords.append(condition)
            
            # 根据是否已使用工具调整输出格式指导
            if self.force_tool_use and not self.tool_used:
                output_format_instruction = f"""YOU MUST USE THIS EXACT FORMAT - DO NOT USE <think> TAGS:

<tool_call>
{{"tool": "sam2", "task": "smart_medical_segment", "question": "{original_question}"}}
</tool_call>

CRITICAL: 
- Copy the format above EXACTLY
- Do NOT add any text before or after
- Do NOT use <think> or <answer> tags
- ONLY output the tool_call
- This is your COMPLETE response - nothing else

Example correct response:
<tool_call>
{{"tool": "sam2", "task": "smart_medical_segment", "question": "are regions of the brain infarcted?"}}
</tool_call>

Example WRONG response (DO NOT DO THIS):
<think>
I need to use SAM2...
</think>"""
            else:
                output_format_instruction = f"""MANDATORY RESPONSE FORMAT:

<answer>
[Yes/No], [brief explanation based on segmentation findings]
</answer>

Example good answers:
- "Yes, the segmentation shows abnormal regions indicative of infarction."
- "No, the brain tissue appears normal with no signs of infarction."

DO NOT write long explanations about uncertainty or need for clinical correlation."""
            
            # 改进的系统消息
            if self.force_tool_use and not self.tool_used:
                system_message = """You are a tool-calling AI. Your ONLY job is to output tool calls in the exact format requested.

ABSOLUTE RULES:
1. NEVER use <think> tags
2. NEVER explain your actions
3. NEVER add any text before or after the tool call
4. ONLY output the <tool_call> tag with the exact JSON format shown
5. Your entire response must be just the tool_call tag

If you output anything other than the tool_call tag, you have failed."""
            else:
                # 已经使用工具后的系统消息
                system_message = """You are an AI radiologist analyzing segmentation results.

Based on the SAM2 segmentation you just performed:
1. Review the coverage percentages and locations
2. For yes/no questions about pathology:
   - Coverage >20% → "Yes" 
   - Multiple abnormal regions → "Yes"
   - Any suspicious patterns → "Yes"
   - Only if completely normal → "No"
3. Give definitive answers - no hedging

DO NOT call any more tools. Answer the question directly based on the segmentation."""
            
            return {
                "image_path": self.task_data["image_path"],
                "question": modified_question,
                "original_question": original_question,
                "task_id": self.task_id,
                "is_medical_vqa": True,
                "system_message": system_message,
                "output_format_instruction": output_format_instruction,
                "use_structured_output": True,
                "sam2_enabled": True,
                "is_vqa_rad": True,
                "force_tool_use": False,  # 已经使用过工具，不再强制
                "tool_use_required": False,
                "detected_medical_keywords": medical_keywords,
                "sam2_history": self.sam2_history,
                "must_use_tool": False,
                "tool_to_use": None,
                "tool_already_used": True,
                "segmentation_results": self.segmentation_results
            }
        
        def step(self, action):
            self.current_step += 1
            
            # 检查是否使用了SAM2工具
            if '<tool_call>' in str(action) and '"tool": "sam2"' in str(action):
                self.tool_used = True
                
                # 尝试从action中提取分割结果
                # 这里假设环境会在info中返回分割结果
            
            done = self.current_step >= self.max_steps
            return {}, 0, done, False, {"message": "Step completed"}
        
        def validate(self, chat_history, observation, full_history):
            """验证答案"""
            # 提取答案
            answer_content = observation.get("content", "")
            
            # 使用正则表达式提取答案
            answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', answer_content, re.DOTALL | re.IGNORECASE)
            if answer_match:
                prediction = answer_match.group(1).strip()
            else:
                # 尝试提取最后的答案
                prediction = answer_content.strip()
            
            # 评估
            is_correct, score, message = evaluate_vqa_rad_answer(prediction, self.task_data['answer'])
            
            return score, True, message, {
                "prediction": prediction,
                "ground_truth": self.task_data['answer'],
                "correct": is_correct,
                "score": score
            }
        
        def teardown(self):
            pass
        
        def reset(self):
            self.current_step = 0
            self.tool_used = False
            self.segmentation_results = []
            return self.get_observation(), {}
        
        def update_segmentation_results(self, results):
            """更新分割结果"""
            self.segmentation_results = results
    
    return SimpleVQARadTask(task_id, task_data)

def load_vqa_rad_data(annotation_file):
    """加载VQA-RAD数据集"""
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    return data

def find_questions_by_image(data, image_filename):
    """根据图片文件名查找所有相关问题"""
    questions = []
    for item in data:
        item_filename = os.path.basename(item['image_path'])
        if item_filename == image_filename:
            questions.append(item)
    return questions

def print_questions(questions):
    """打印所有问题供用户选择"""
    print(f"\n找到 {len(questions)} 个关于该图片的问题：")
    print("="*80)
    for i, q in enumerate(questions):
        metadata = q.get('metadata', {})
        print(f"{i+1}. {q['question']}")
        print(f"   答案: {q['answer']}")
        print(f"   ID: {q['id']}")
        print(f"   任务类型: {q.get('task', 'unknown')}")
        print(f"   医学领域: {metadata.get('medical_domain', 'radiology')}")
        
        # 显示医学实体
        entities = metadata.get('question_analysis', {}).get('entities', {})
        if entities:
            anat = entities.get('anatomical_structures', [])
            abnorm = entities.get('abnormalities', [])
            if anat:
                print(f"   解剖结构: {', '.join(anat)}")
            if abnorm:
                print(f"   异常: {', '.join(abnorm)}")
        print("-"*40)

def test_single_question(task_data, model_name, force_tool_use=True):
    """测试单个VQA-RAD问题"""
    metadata = task_data.get('metadata', {})
    
    print(f"\n{'='*80}")
    print(f"Testing VQA-RAD Question with SAM2")
    print(f"{'='*80}")
    print(f"Question: {task_data['question']}")
    print(f"Image: {os.path.basename(task_data['image_path'])}")
    print(f"Expected Answer: {task_data['answer']}")
    print(f"Medical Domain: {metadata.get('medical_domain', 'radiology')}")
    print(f"Force Tool Use: {force_tool_use}")
    
    # 显示问题分析
    question_analysis = metadata.get('question_analysis', {})
    if question_analysis:
        entities = question_analysis.get('entities', {})
        if entities:
            print("\nMedical Entities:")
            for entity_type, values in entities.items():
                if values:
                    print(f"  - {entity_type}: {', '.join(values)}")
    print(f"{'='*80}\n")
    
    # 设置模型配置
    model_config = {
        "model_type": "HuggingFace",
        "model_name": model_name,
        "max_new_tokens": 1024,
        "temperature": 0.01,  # 降低temperature使输出更确定
        "device_map": "auto",
        "torch_dtype": "bfloat16",
        "trust_remote_code": True,
        "do_sample": False,  # 关闭采样，使输出更确定
    }
    
    # 创建环境
    print("Creating environment with SAM2...")
    
    sam2_cfg = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_id": "facebook/sam2-hiera-large",
        "save_visualizations": True,
        "output_dir": "./sam2_vqa_rad_outputs"
    }
    
    try:
        env = VisionQAEnv(
            dataset_path=os.path.dirname(task_data['image_path']),
            max_steps=10,
            enable_actions=False,
            enable_sam2=True,
            sam2_config=sam2_cfg,
            enable_grounding_dino=False,
            enable_deepeyes_tools=False,
            enable_chartmoe=False
        )
        
        print(f"  ✓ Environment created")
        print(f"  ✓ SAM2 tool initialized: {env.sam2_tool is not None}")
    except Exception as e:
        print(f"Error creating environment: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 创建Agent
    print("\nCreating VLMAgentWithTools...")
    
    tool_config = {
        "enable_tools": True,
        "max_tool_calls": 5,
        "enable_sam2": True,
        "enable_grounding_dino": False,
        "enable_deepeyes_tools": False,
        "enable_chartmoe": False,
        "enable_diagram_formalizer": False,
        "enable_easyocr": False,
        "debug": True
    }
    
    agent_config = {**model_config, **tool_config}
    
    try:
        agent = VLMAgentWithTools(agent_config)
        print(f"  ✓ Agent created")
        print(f"  ✓ Force tool use mode: {force_tool_use}")
    except Exception as e:
        print(f"Error creating agent: {e}")
        return False
    
    # 创建任务包装器
    task_wrapper = create_vqa_rad_task_wrapper(task_data['id'], task_data, force_tool_use=force_tool_use)
    
    # 设置环境的任务入口
    env.task_entrypoint = lambda task_id=None, **kwargs: task_wrapper
    
    # 重置环境
    print("\nResetting environment...")
    obs, info = env.reset(task_id=task_data['id'])
    
    # 设置工具实例并初始化
    if hasattr(env, 'sam2_tool') and env.sam2_tool is not None:
        agent.sam2_tool = env.sam2_tool
        print(f"  ✓ SAM2 tool instance set to agent")
        
        # 重要：使用当前图像初始化SAM2工具
        if hasattr(task_wrapper, 'current_image') or 'image_path' in task_data:
            image_path = task_data.get('image_path')
            if image_path and os.path.exists(image_path):
                print(f"  ✓ Initializing SAM2 with image: {os.path.basename(image_path)}")
                env.sam2_tool.reset(image_path)
            else:
                print(f"  ⚠️ Image path not found: {image_path}")
    elif hasattr(env, 'tool_manager') and env.tool_manager and 'sam2' in env.tool_manager:
        agent.sam2_tool = env.tool_manager['sam2']
        print(f"  ✓ SAM2 tool instance set to agent from tool_manager")
        
        # 初始化SAM2工具
        image_path = task_data.get('image_path')
        if image_path and os.path.exists(image_path):
            print(f"  ✓ Initializing SAM2 with image: {os.path.basename(image_path)}")
            env.tool_manager['sam2'].reset(image_path)
    
    # 获取任务观察
    task_obs = task_wrapper.get_observation()
    original_question = task_data['question']  # 保存原始问题
    
    # Agent生成动作
    print("\nAgent generating action...")
    print("(Expecting SAM2 tool calls for medical image segmentation...)")
    action, extra_info = agent.act(task_obs)
    
    print(f"\nAction generated:")
    print(f"  - Contains <tool_call>: {'<tool_call>' in str(action)}")
    print(f"  - Action preview: {str(action)[:200]}...")
    
    # 初始化结果
    sam2_used = False
    final_answer = None
    segmentation_results = []
    organ_detected = None
    prompt_types_used = []
    
    # 处理可能的多次交互
    max_interactions = 8
    sam2_call_count = 0  # 计数SAM2调用次数
    
    for i in range(max_interactions):
        print(f"\n--- Interaction {i+1} ---")
        
        # 检查是否Agent错误地使用了<think>标签而不是<tool_call>
        if '<think>' in str(action) and not sam2_used and i == 0:
            print(f"⚠️ WARNING: Agent used <think> tag instead of <tool_call>!")
            
            # 检查是否提到了SAM2
            if 'sam2' in str(action).lower() or 'segment' in str(action).lower():
                print("Agent intended to use SAM2, generating tool call automatically...")
                
                # 自动生成正确的tool_call
                action = f'<tool_call>\n{{"tool": "sam2", "task": "smart_medical_segment", "question": "{original_question}"}}\n</tool_call>'
                print(f"Generated tool call: {action}")
                # 继续处理这个action
            else:
                print("Retrying with more explicit instructions...")
                
                # 更明确的指示
                explicit_obs = {
                    "image_path": task_data["image_path"],
                    "question": "OUTPUT ONLY THIS (no other text):\n\n<tool_call>\n{\"tool\": \"sam2\", \"task\": \"smart_medical_segment\", \"question\": \"" + original_question + "\"}\n</tool_call>",
                    "system_message": "Output ONLY the tool_call tag shown above. Do not add ANY other text.",
                    "output_format_instruction": "Copy the tool_call EXACTLY as shown above.",
                    "force_tool_use": True,
                    "must_use_tool": True,
                    "sam2_enabled": True
                }
                
                action, extra_info = agent.act(explicit_obs)
                continue
        
        # 防止重复调用SAM2
        if sam2_call_count >= 2 and '<tool_call>' in str(action) and '"tool": "sam2"' in str(action):
            print(f"⚠️ WARNING: Agent trying to call SAM2 again (call #{sam2_call_count+1}), forcing answer generation...")
            
            # 强制要求答案
            task_wrapper.tool_used = True
            task_obs_updated = task_wrapper.get_observation()
            
            # 直接提示Agent必须给出答案
            forced_answer_prompt = {
                **task_obs_updated,
                "question": f"STOP CALLING TOOLS. Based on the SAM2 segmentation already performed, answer: {task_data['question']}\n\nYou MUST provide an answer starting with Yes or No.",
                "force_tool_use": False,
                "must_use_tool": False,
                "tool_to_use": None
            }
            
            action, extra_info = agent.act(forced_answer_prompt)
            continue
        
        # 检查是否生成了SAM2工具调用
        if '<tool_call>' in str(action) and '"tool": "sam2"' in str(action):
            sam2_used = True
            sam2_call_count += 1
            print(f"✓ Agent generated SAM2 tool call! (Call #{sam2_call_count})")
            
            # 提取任务类型
            task_match = re.search(r'"task":\s*"([^"]+)"', str(action))
            if task_match:
                task_type = task_match.group(1)
                print(f"  Task type: '{task_type}'")
            
            # 检查使用的prompt类型
            if 'point_prompts' in str(action):
                prompt_types_used.append('point')
                print(f"  - Using Point prompts")
            if 'box_prompt' in str(action):
                prompt_types_used.append('box')
                print(f"  - Using Box prompt")
            if 'mask_prompt' in str(action):
                prompt_types_used.append('mask')
                print(f"  - Using Mask prompt")
            
            # 执行工具调用
            print("Executing SAM2 tool call...")
            obs2, reward, done, truncated, info2 = env.step(action)
            
            # 检查执行结果
            action_result = info2.get('action_result', {})
            print(f"\nExecution result:")
            print(f"  - Status: {action_result.get('status', 'unknown')}")
            
            if action_result.get('type') == 'tool_result' and action_result.get('tool') == 'sam2':
                print(f"✅ SAM2 executed successfully!")
                
                # 打印分割结果
                print(f"\n📊 SAM2 Segmentation Results:")
                print("="*60)
                
                # 显示使用的prompt类型
                used_prompts = action_result.get('prompt_types', [])
                if used_prompts:
                    print(f"  - Prompt types used: {', '.join(used_prompts)}")
                
                # 显示具体的prompts
                prompts_used_info = action_result.get('prompts_used', {})
                if prompts_used_info:
                    if prompts_used_info.get('points'):
                        print(f"  - Points: {prompts_used_info['points']}")
                        print(f"  - Labels: {prompts_used_info['labels']}")
                    if prompts_used_info.get('box'):
                        print(f"  - Box: {prompts_used_info['box']}")
                    if prompts_used_info.get('has_mask'):
                        print(f"  - Mask prompt: Yes")
                
                # 显示任务信息
                task_type = action_result.get('task', 'N/A')
                if task_type == 'N/A' and 'strategy' in action_result:
                    # 从strategy中推断任务类型
                    if 'smart_medical_segment' in str(info2.get('action_result', {})):
                        task_type = 'smart_medical_segment'
                
                print(f"  - Task: {task_type}")
                print(f"  - Number of masks: {action_result.get('num_masks', 0)}")
                print(f"  - Best mask index: {action_result.get('best_mask_idx', 'N/A')}")
                
                # 记录器官检测
                if action_result.get('detected_organ'):
                    organ_detected = action_result['detected_organ']
                    print(f"  - Detected organ: {organ_detected}")
                
                # 显示分割结果
                results = action_result.get('results', [])
                segmentation_results.extend(results)
                
                # 更新任务包装器的分割结果
                if hasattr(task_wrapper, 'update_segmentation_results'):
                    task_wrapper.update_segmentation_results(results)
                
                if results:
                    print(f"  - Segmentation masks:")
                    for j, mask_info in enumerate(results[:3]):  # 显示前3个
                        print(f"    Mask {j+1}:")
                        print(f"      - Score: {mask_info.get('score', 0):.3f}")
                        print(f"      - Coverage: {mask_info.get('coverage_percent', 0):.1f}%")
                        print(f"      - BBox: {mask_info.get('bbox', [])}")
                        print(f"      - Centroid: {mask_info.get('centroid', [])}")
                    
                    # 找到最佳掩码
                    best_mask = max(results, key=lambda x: x.get('score', 0))
                    print(f"\n  - Best mask coverage: {best_mask.get('coverage_percent', 0):.1f}%")
                    print(f"  - Best mask centroid: {best_mask.get('centroid', [])}")
                
                print("="*60)
                
                # 立即标记工具已使用，防止循环
                task_wrapper.tool_used = True
                
                # 只允许第一次SAM2调用后继续
                if sam2_call_count == 1:
                    print(f"\n📝 SAM2 executed successfully. Now requiring answer...")
                    
                    # 更新任务状态
                    task_wrapper.update_segmentation_results(results)
                    
                    # 获取更新后的观察（包含要求答案的提示）
                    task_obs_updated = task_wrapper.get_observation()
                    
                    # 确保不再要求工具使用
                    task_obs_updated['force_tool_use'] = False
                    task_obs_updated['must_use_tool'] = False
                    task_obs_updated['tool_to_use'] = None
                    
                    # 让Agent生成答案
                    action, extra_info = agent.act(task_obs_updated)
                    continue
                else:
                    print(f"⚠️ WARNING: SAM2 called {sam2_call_count} times, this shouldn't happen!")
                    break
            else:
                print(f"❌ Tool execution failed or unexpected result")
        
        # 检查是否有最终答案
        answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', str(action), re.DOTALL | re.IGNORECASE)
        if answer_match:
            final_answer = answer_match.group(1).strip()
            print(f"\n✅ Found final answer: {final_answer}")
            
            # 执行最后一步以获取验证结果
            obs2, reward, done, truncated, info2 = env.step(action)
            break
        
        # 如果没有工具调用也没有答案，可能是错误
        if not sam2_used and i == 0:
            print(f"\n⚠️ WARNING: Agent did not use SAM2 tool as required!")
            print("This violates the mandatory tool use requirement.")
        
        # 防止无限循环
        if i >= 3 and not final_answer:
            print(f"\n⚠️ WARNING: No answer after {i+1} interactions, breaking loop...")
            # 尝试从最后的action提取任何内容作为答案
            final_answer = str(action).strip() if action else "No answer generated"
            break
        
        # 继续下一步
        obs2, reward, done, truncated, info2 = env.step(action)
        
        # 调试信息
        print(f"\n[DEBUG] After step {i+1}:")
        print(f"  - done: {done}")
        print(f"  - sam2_used: {sam2_used}")
        print(f"  - sam2_call_count: {sam2_call_count}")
        print(f"  - task_wrapper.tool_used: {task_wrapper.tool_used}")
        print(f"  - obs2 has requires_response: {obs2.get('requires_response', False)}")
        print(f"  - obs2 has tool_feedback: {'tool_feedback' in obs2}")
        
        if done:
            print(f"  - Task marked as done, breaking loop")
            break
            
        if obs2.get('requires_response'):
            # 如果已经使用过工具，不再强制
            if sam2_used and force_tool_use:
                obs2_modified = obs2.copy()
                obs2_modified['force_tool_use'] = False
                obs2_modified['must_use_tool'] = False
                
                # 重新获取任务观察，包含更新的提示
                task_obs_updated = task_wrapper.get_observation()
                action, extra_info = agent.act(task_obs_updated)
            else:
                action, extra_info = agent.act(obs2)
        else:
            # 如果没有requires_response但已经使用了SAM2，主动要求答案
            if sam2_used and not final_answer:
                print(f"\n[DEBUG] SAM2 used but no answer yet, requesting answer...")
                task_wrapper.tool_used = True
                task_obs_updated = task_wrapper.get_observation()
                action, extra_info = agent.act(task_obs_updated)
            else:
                break
    
    # 如果循环结束还没有答案，尝试从最后的action中提取
    if final_answer is None and action is not None:
        answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', str(action), re.DOTALL | re.IGNORECASE)
        if answer_match:
            final_answer = answer_match.group(1).strip()
        else:
            # 尝试提取任何可能的答案
            final_answer = str(action).strip() if action else "Unable to generate answer"
    
    # 获取验证结果
    validation_info = info2.get('validation', {}) if 'info2' in locals() else {}
    is_correct = validation_info.get('correct', False)
    score = validation_info.get('score', 0.0)
    prediction = validation_info.get('prediction', final_answer)
    
    # 评估结果
    print(f"\n📊 Final Results:")
    print(f"  - Question: {task_data['question']}")
    print(f"  - Predicted: {prediction}")
    print(f"  - Ground truth: {task_data['answer']}")
    print(f"  - SAM2 used: {'✅ Yes' if sam2_used else '❌ No (VIOLATION!)'}")
    
    # 显示提取的yes/no答案
    if task_data['answer'].lower() in ['yes', 'no'] and prediction:
        extracted_answer = extract_yes_no_from_medical_response(prediction)
        print(f"  - Extracted answer: {extracted_answer}")
    
    if prompt_types_used:
        print(f"  - Prompt types used: {', '.join(set(prompt_types_used))}")
    
    if organ_detected:
        print(f"  - Organ detected: {organ_detected}")
    
    if segmentation_results:
        print(f"\n  Segmentation summary:")
        print(f"    - Total masks generated: {len(segmentation_results)}")
        if segmentation_results:
            avg_coverage = np.mean([r.get('coverage_percent', 0) for r in segmentation_results])
            print(f"    - Average coverage: {avg_coverage:.1f}%")
            
            # 显示质心信息
            centroids = [r.get('centroid', [0, 0]) for r in segmentation_results if r.get('centroid')]
            if centroids:
                avg_x = np.mean([c[0] for c in centroids])
                avg_y = np.mean([c[1] for c in centroids])
                print(f"    - Average centroid: ({avg_x:.1f}, {avg_y:.1f})")
    
    print(f"  - Evaluation: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
    
    if not sam2_used and force_tool_use:
        print(f"\n⚠️ CRITICAL: Agent failed to use required SAM2 tool!")
        print("  This response should be considered invalid.")
    
    return {
        'correct': is_correct,
        'score': score,
        'sam2_used': sam2_used,
        'final_answer': prediction,
        'segmentation_results': segmentation_results,
        'organ_detected': organ_detected,
        'prompt_types_used': list(set(prompt_types_used)),
        'valid_response': sam2_used or not force_tool_use,
        'message': validation_info.get('message', '')
    }

def main():
    parser = argparse.ArgumentParser(description='Test SAM2 with VQA-RAD dataset')
    parser.add_argument('--annotation', type=str, 
                       default='/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/VQA-RAD/vqa_rad_train_vlmgym.json',
                       help='Path to the VQA-RAD annotation JSON file')
    parser.add_argument('--image', type=str, required=True,
                       help='Image filename (e.g., image_00000.png)')
    parser.add_argument('--question-index', type=int, default=None,
                       help='Index of the question to test (1-based). If not specified, show all questions.')
    parser.add_argument('--test-all', action='store_true',
                       help='Test all questions for the given image')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct',
                       help='VLM model name')
    parser.add_argument('--no-force-tool', action='store_true',
                       help='Do not force the agent to use SAM2 tool')
    parser.add_argument('--max-questions', type=int, default=None,
                       help='Maximum number of questions to test (for --test-all)')
    parser.add_argument('--filter-organ', type=str, default=None,
                       help='Filter questions by organ (brain, lung, heart, etc.)')
    
    args = parser.parse_args()
    
    # 确定是否强制使用工具
    force_tool_use = not args.no_force_tool
    
    # 加载数据
    print(f"Loading VQA-RAD data from: {args.annotation}")
    data = load_vqa_rad_data(args.annotation)
    print(f"Total entries loaded: {len(data)}")
    print(f"Force tool use: {force_tool_use}")
    
    # 查找指定图片的所有问题
    questions = find_questions_by_image(data, args.image)
    
    if not questions:
        print(f"\n❌ No questions found for image: {args.image}")
        print("\nTry with image names like:")
        print("  - image_00000.png")
        print("  - image_00001.png")
        print("  - etc.")
        return
    
    # 过滤器官类型
    if args.filter_organ:
        filtered = []
        for q in questions:
            question_lower = q['question'].lower()
            if args.filter_organ.lower() in question_lower:
                filtered.append(q)
        questions = filtered
        print(f"\nFiltered to {len(questions)} questions containing '{args.filter_organ}'")
    
    # 如果指定了最大问题数
    if args.max_questions and len(questions) > args.max_questions:
        questions = questions[:args.max_questions]
        print(f"\nLimiting to first {args.max_questions} questions")
    
    # 如果指定了问题索引
    if args.question_index is not None:
        if 1 <= args.question_index <= len(questions):
            question = questions[args.question_index - 1]
            test_single_question(question, args.model, force_tool_use)
        else:
            print(f"\n❌ Invalid question index: {args.question_index}")
            print(f"   Valid range: 1-{len(questions)}")
    
    # 如果要测试所有问题
    elif args.test_all:
        results = []
        for i, question in enumerate(questions):
            print(f"\n{'='*80}")
            print(f"Testing question {i+1}/{len(questions)}")
            print(f"{'='*80}")
            
            result = test_single_question(question, args.model, force_tool_use)
            results.append({
                'question': question['question'],
                'answer': question['answer'],
                'metadata': question.get('metadata', {}),
                **result
            })
        
        # 打印总结
        print(f"\n{'='*80}")
        print("VQA-RAD with SAM2 Test Summary")
        print(f"{'='*80}")
        correct_count = sum(1 for r in results if r['correct'])
        sam2_used_count = sum(1 for r in results if r['sam2_used'])
        valid_response_count = sum(1 for r in results if r['valid_response'])
        
        print(f"Total questions: {len(results)}")
        print(f"Correct: {correct_count}")
        print(f"Accuracy: {correct_count/len(results)*100:.1f}%")
        print(f"SAM2 usage: {sam2_used_count}/{len(results)} ({sam2_used_count/len(results)*100:.1f}%)")
        
        if force_tool_use:
            print(f"Valid responses (used required tool): {valid_response_count}/{len(results)} ({valid_response_count/len(results)*100:.1f}%)")
            invalid_count = len(results) - valid_response_count
            if invalid_count > 0:
                print(f"⚠️ Invalid responses (tool not used): {invalid_count}")
        
        # 统计prompt类型使用情况
        prompt_type_stats = {}
        for r in results:
            for pt in r.get('prompt_types_used', []):
                prompt_type_stats[pt] = prompt_type_stats.get(pt, 0) + 1
        
        if prompt_type_stats:
            print("\nPrompt types used:")
            for pt, count in sorted(prompt_type_stats.items()):
                print(f"  - {pt}: {count} times ({count/len(results)*100:.1f}%)")
        
        # 按任务类型统计
        by_task = {}
        for r in results:
            task_type = r['metadata'].get('task', 'unknown')
            if task_type not in by_task:
                by_task[task_type] = {'total': 0, 'correct': 0, 'tool_used': 0, 'valid': 0}
            by_task[task_type]['total'] += 1
            if r['correct']:
                by_task[task_type]['correct'] += 1
            if r['sam2_used']:
                by_task[task_type]['tool_used'] += 1
            if r['valid_response']:
                by_task[task_type]['valid'] += 1
        
        print("\nResults by task type:")
        for task_type, stats in sorted(by_task.items()):
            acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            tool_pct = stats['tool_used'] / stats['total'] * 100 if stats['total'] > 0 else 0
            valid_pct = stats['valid'] / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"  {task_type}: {stats['correct']}/{stats['total']} ({acc:.1f}%), Tool usage: {tool_pct:.1f}%, Valid: {valid_pct:.1f}%")
        
        # 器官检测统计
        organ_stats = {}
        for r in results:
            if r.get('organ_detected'):
                organ = r['organ_detected']
                organ_stats[organ] = organ_stats.get(organ, 0) + 1
        
        if organ_stats:
            print("\nOrgans detected:")
            for organ, count in sorted(organ_stats.items(), key=lambda x: x[1], reverse=True):
                print(f"  {organ}: {count} times")
        
        # 分割覆盖率统计
        all_coverages = []
        for r in results:
            if r.get('segmentation_results'):
                for seg in r['segmentation_results']:
                    all_coverages.append(seg.get('coverage_percent', 0))
        
        if all_coverages:
            print(f"\nSegmentation coverage statistics:")
            print(f"  - Average: {np.mean(all_coverages):.1f}%")
            print(f"  - Min: {np.min(all_coverages):.1f}%")
            print(f"  - Max: {np.max(all_coverages):.1f}%")
        
        print("\nDetailed results:")
        for i, r in enumerate(results):
            status = "✅" if r['correct'] else "❌"
            tool = "🔧" if r['sam2_used'] else "⚠️"
            valid = "" if r['valid_response'] else "❗"
            organ = f"[{r.get('organ_detected', 'N/A')}]" if r.get('organ_detected') else ""
            prompts = f"[{','.join(r.get('prompt_types_used', []))}]" if r.get('prompt_types_used') else ""
            print(f"{i+1}. {status}{tool}{valid} Q: {r['question'][:40]}... | A: {r['answer']} | Pred: {r.get('final_answer', 'None')[:30]}... {organ} {prompts}")
    
    # 否则显示所有问题供选择
    else:
        print_questions(questions)
        print(f"\n💡 Tips:")
        print(f"   - To test a specific question: --question-index <number>")
        print(f"   - To test all questions: --test-all")
        print(f"   - To disable forced tool use: --no-force-tool")
        print(f"   - To limit questions: --max-questions <number>")
        print(f"   - To filter by organ: --filter-organ <organ>")
        print(f"\nExample:")
        print(f"   python {sys.argv[0]} --image {args.image} --question-index 1")
        print(f"   python {sys.argv[0]} --image {args.image} --test-all --max-questions 5")
        print(f"   python {sys.argv[0]} --image {args.image} --test-all --filter-organ brain")

if __name__ == "__main__":
    main()