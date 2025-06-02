#!/bin/bash
set -e

echo "========================================="
echo "VLM Gym Docker Container Started"
echo "========================================="
echo "Task: unified evaluation"
echo "Model: ${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
echo "Dataset: ${DATASET:-all}"
echo "Samples: ${NUM_SAMPLES:-5}"
echo "========================================="

cd /workspace

# 运行评估脚本
python scripts/run_unified_eval.py \
    --annotation /workspace/data/vision_r1_sample_dataset.json \
    --data-root /workspace/data \
    --dataset ${DATASET:-all} \
    --model ${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct} \
    --limit ${NUM_SAMPLES:-5} \
    --save-conversations
