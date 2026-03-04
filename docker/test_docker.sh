#!/bin/bash
cd "$(dirname "$0")/.."

echo "Running quick test..."
docker run \
    --rm \
    --name vlm_gym_test \
    --gpus="device=${GPU_DEVICE:-0}" \
    --shm-size=32g \
    -v $(pwd):/workspace \
    -v ${HOME}/.cache/huggingface:/workspace/cache \
    -it vlm_gym:latest \
    python scripts/run_chartqa_eval_reflection_with_tool.py \
        --annotation data/chartqa/converted_train/train_human_vlmgym_container.json \
        --data-root data/chartqa \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --enable-reflection \
        --max-attempts 3 \
        --limit 2
