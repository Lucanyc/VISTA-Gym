import re


def extract_boxed_content(text: str) -> str:
    pattern = r"\\boxed\{([^}]*)\}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return ""


def normalize_answer(answer: str) -> str:
    answer = answer.strip().upper()
    # 只保留选项字母
    if answer in ['A', 'B', 'C', 'D']:
        return answer
    # 尝试提取首字母
    match = re.match(r'^([A-D])[.\s\)]', answer)
    if match:
        return match.group(1)
    return answer


def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    return 1.0 if re.fullmatch(pattern, predict_str) else 0.0


def acc_reward(predict_str: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict_str)
    if not answer:
        return 0.0
    return 1.0 if normalize_answer(answer) == normalize_answer(ground_truth) else 0.0


def compute_score(predict_str: str, ground_truth: str, format_score: float = 0.1) -> float:
    boxed_gt = extract_boxed_content(ground_truth)
    if boxed_gt:
        ground_truth = boxed_gt

    return (1.0 - format_score) * acc_reward(predict_str, ground_truth) + format_score * format_reward(predict_str)