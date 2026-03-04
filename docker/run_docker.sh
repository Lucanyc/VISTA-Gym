#!/bin/bash
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Default parameters (override with environment variables)
GPU_DEVICE=${GPU_DEVICE:-"0"}

echo "Starting VLM Gym Docker container..."
docker run \
    --rm \
    --name vlm_gym_eval \
    --gpus="device=${GPU_DEVICE}" \
    --shm-size=32g \
    -v ${PROJECT_ROOT}:/workspace \
    -v ${HOME}/.cache/huggingface:/workspace/cache \
    -e CUDA_VISIBLE_DEVICES="${GPU_DEVICE}" \
    -it vlm_gym:latest
