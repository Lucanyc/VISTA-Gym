# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
CLEVR dataset reward function for verl GRPO training.

CLEVR 答案类型包括：
- 数量: 0, 1, 2, 3, ...
- 颜色: gray, red, blue, green, brown, purple, cyan, yellow
- 形状: cube, sphere, cylinder
- 尺寸: small, large
- 材质: metal, rubber (shiny, matte)
- 布尔: yes, no
"""
import re


def extract_boxed_content(text: str) -> str:
    """从 \\boxed{...} 中提取答案内容"""
    # 匹配最后一个 \boxed{...}
    pattern = r"\\boxed\{([^}]*)\}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return ""


def normalize_answer(answer: str) -> str:
    """
    标准化答案，用于精确匹配比较。
    - 转小写
    - 去除首尾空格和标点
    - 统一常见同义表达
    """
    answer = answer.strip().lower()
    # 去除末尾标点
    answer = re.sub(r"[。.，,;；!！?？\s]+$", "", answer)
    # 去除首尾引号
    answer = answer.strip("\"'""''")

    # 统一同义词映射（根据 CLEVR 数据集特点）
    synonym_map = {
        "true": "yes",
        "false": "no",
        "correct": "yes",
        "incorrect": "no",
        "right": "yes",
        "wrong": "no",
        # 材质同义
        "shiny": "metal",
        "matte": "rubber",
        "metallic": "metal",
        # 形状同义
        "block": "cube",
        "ball": "sphere",
    }

    if answer in synonym_map:
        answer = synonym_map[answer]

    return answer


def format_reward(predict_str: str) -> float:
    """
    检查模型输出是否符合 <think>...</think> ... \\boxed{...} 格式。
    符合格式给 1.0，否则给 0.0。
    """
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


def acc_reward(predict_str: str, ground_truth: str, use_boxed: bool = True) -> float:
    """
    精确匹配奖励。
    从模型输出中提取答案，与 ground_truth 做标准化后的精确匹配。
    匹配返回 1.0，否则返回 0.0。
    """
    if use_boxed:
        answer = extract_boxed_content(predict_str)
    else:
        answer = predict_str

    if not answer:
        return 0.0

    normalized_pred = normalize_answer(answer)
    normalized_gt = normalize_answer(ground_truth)

    # 精确匹配
    if normalized_pred == normalized_gt:
        return 1.0

    # 数字容错：尝试数值比较（处理 "3" vs "3.0" 这类情况）
    try:
        if float(normalized_pred) == float(normalized_gt):
            return 1.0
    except (ValueError, TypeError):
        pass

    return 0.0


def compute_score(predict_str: str, ground_truth: str, use_boxed: bool = True, format_score: float = 0.1) -> float:
    # 如果 ground_truth 本身带 \boxed{}，先提取出来
    boxed_gt = extract_boxed_content(ground_truth)
    if boxed_gt:
        ground_truth = boxed_gt

    return (1.0 - format_score) * acc_reward(predict_str, ground_truth, use_boxed) + format_score * format_reward(
        predict_str
    )