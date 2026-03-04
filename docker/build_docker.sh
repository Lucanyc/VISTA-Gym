#!/bin/bash
cd "$(dirname "$0")/.."
echo "Building VLM Gym Docker image..."
docker build -f docker/Dockerfile -t vlm_gym:latest .
echo "Build complete!"
