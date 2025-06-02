#!/bin/bash

# set root
PROJECT_ROOT="/data/wang/meng/GYM-Work/try_vlm_gym"

# set parameters
DATASET=${DATASET:-"all"}
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-VL-7B-Instruct"}
NUM_SAMPLES=${NUM_SAMPLES:-"5"}
GPU_DEVICE=${GPU_DEVICE:-"0"}

echo "Starting VLM Gym Docker container..."

docker run \
    --rm \
    --name vlm_gym_eval \
    --gpus="device=${GPU_DEVICE}" \
    --shm-size=32g \
    -v ${PROJECT_ROOT}:/workspace:ro \
    -v ${PROJECT_ROOT}/experiments:/workspace/experiments \
    -v ${PROJECT_ROOT}/workdir:/workspace/workdir \
    -v ${HOME}/.cache/huggingface:/workspace/cache \
    -e DATASET="${DATASET}" \
    -e MODEL_NAME="${MODEL_NAME}" \
    -e NUM_SAMPLES="${NUM_SAMPLES}" \
    -e CUDA_VISIBLE_DEVICES="${GPU_DEVICE}" \
    -it vlm_gym:latest
