#!/bin/bash

cd /data/wang/meng/GYM-Work/try_vlm_gym

echo "Building VLM Gym Docker image..."
docker build -f docker/Dockerfile -t vlm_gym:latest .
echo "Build complete!"
