# Scaling Agentic Reinforcement Learning for Tool-Integrated Reasoning in VLMs

📑 **Contents**
- [📖 Overview](#-overview)
- [⚙️ Installation](#-installation)
- [🗺️ VISTA-Gym](#-vista-gym)
- [🚀 Full Training Pipeline](#-full-training-pipeline)
  - [🧠 Step 1: Supervised Fine-Tuning](#-step-1-supervised-fine-tuning)
  - [🎯 Step 2: Reinforcement Learning (GRPO)](#-step-2-reinforcement-learning-grpo)
- [🏆 Project Info](#-project-info)

---

## 📖 Overview

While recent vision-language models (VLMs) demonstrate strong image understanding, their ability to "think with images" — to reason through multi-step visual interactions — remains limited.

We introduce **VISTA-Gym**, a scalable training environment for incentivizing tool-integrated visual reasoning capabilities in VLMs. VISTA-Gym unifies diverse real-world multimodal reasoning tasks (7 tasks from 13 datasets in total) with a standardized interface for visual tools (e.g., grounding, parsing), executable interaction loops, verifiable feedback signals, and efficient trajectory logging, enabling visual agentic reinforcement learning at scale. With VISTA-Gym, we train **VISTA-R1** to interleave tool-use with agentic reasoning via multi-turn trajectory sampling and end-to-end reinforcement learning. 

![VISTA Overview](./assets/figure1.png)

---

## ⚙️ Installation

### Gym setup

```bash
git clone https://github.com/Lucanyc/vista-gym.git
cd vista-gym
pip install -e .
```

### Docker setup

```bash
cd docker
bash build_docker.sh
bash run_docker.sh
```

### Training setup

We follow the verl/verl-tool environment: [TIGER-AI-Lab/verl-tool](https://github.com/TIGER-AI-Lab/verl-tool/tree/main/verl_tool)

### Tool setup

> *Coming soon*

---

## 🗺️ VISTA-Gym

A scalable reinforcement learning gym for training tool-integrated visual reasoning in VLMs. Drop in any benchmark, any VLM, and evaluate with reflection and tool-augmented reasoning.

**Core components:**

| Directory | Description |
|---|---|
| `vlm_gym/environments/` | Pluggable vision-QA environments (ChartQA, ScienceQA, Geometry3K, etc.) |
| `vlm_gym/agents/` | VLM agent implementations (Qwen2.5-VL-7B-Instruct reference) |
| `vlm_gym/tools/` | Visual tools (OCR, GroundingDINO, image processing) |
| `vlm_gym/tasks/` | Task-specific reasoning and templates |
| `scripts/` | Evaluation entry points |
| `data_adapters/` | Dataset converters to unified vlmgym format |

### Gym Interaction

The gym follows a standard environment-agent loop: the environment sends an observation (image + question), the agent returns an action (predicted answer), and the environment provides feedback for retry.

```
Environment (ChartQA)            Agent (VLM)
    │                                │
    │──── obs: image + question ────►│
    │                                │
    │◄─── action: <think>...</think> │
    │           <answer>Yes</answer> │
    │                                │
    │   [if wrong & reflection on]   │
    │                                │
    │──── feedback + retry ─────────►│
    │◄─── action: revised answer ───│
    │                                │
    │   reward: 1.0 (correct)        │
```

### Data Preparation

Convert ChartQA to vlmgym format:

```bash
python data_adapters/convert_chartqa_to_vlmgym.py
```

### Run ChartQA Evaluation with Reflection

```bash
python scripts/run_chartqa_eval_reflection_with_tool.py \
  --annotation data/chartqa/converted_train/train_human_vlmgym_container.json \
  --data-root data/chartqa \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --enable-reflection \
  --max-attempts 3 \
  --numerical-tolerance 0.05 \
  --limit 1000
```

### Run with Tool-Augmented Reasoning

```bash
python scripts/run_chartqa_eval_reflection_with_tool.py \
  --annotation data/chartqa/converted_train/train_human_vlmgym_container.json \
  --data-root data/chartqa \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --enable-chartqa-reasoning \
  --limit 1000
```

### Quick Test (3 samples)

```bash
python scripts/run_chartqa_eval_reflection_with_tool.py \
  --annotation data/chartqa/converted_train/train_human_vlmgym_container.json \
  --data-root data/chartqa \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --enable-reflection \
  --max-attempts 3 \
  --debug \
  --limit 3
```

### Requirements

- Python 3.10+
- Linux, CUDA 12, NVIDIA GPU (80GB+ recommended for training; inference requires ~20GB for 7B model)
- PyTorch 2.0+
- Transformers 4.40+

---

## 🚀 Full Training Pipeline

We release training code for both **Qwen** (via verl) and **InternVL** families.

### 🧠 Step 1: Supervised Fine-Tuning

> *Coming soon*

### 🎯 Step 2: Reinforcement Learning (GRPO)

> *Coming soon*

---

## 🏆 Project Info

### References

- verl-tool — [TIGER-AI-Lab/verl-tool](https://github.com/TIGER-AI-Lab/verl-tool)
- ChartQA dataset — [ahmed-masry/ChartQA](https://huggingface.co/datasets/ahmed-masry/ChartQA)
- Qwen2.5-VL — [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
