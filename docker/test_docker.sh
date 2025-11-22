#!/bin/bash
cd /your_path/GYM-Work/try_vlm_gym

echo "Running quick test with 2 samples..."
DATASET=chartqa NUM_SAMPLES=2 bash docker/run_docker.sh
