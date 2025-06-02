
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set python3 as default
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /workspace

# Install PyTorch
RUN pip3 install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install requirements with specific numpy version
RUN pip3 install --no-cache-dir \
    numpy==1.26.4 \
    gymnasium>=0.29.0 \
    pandas>=2.0.0 \
    tqdm>=4.65.0 \
    pyyaml>=6.0 \
    matplotlib>=3.5.0 \
    scikit-learn>=1.3.0 \
    Pillow>=9.5.0 \
    opencv-python-headless>=4.8.0 \
    transformers>=4.36.0 \
    accelerate>=0.25.0 \
    qwen-vl-utils \
    einops \
    timm \
    toml \
    jsonlines \
    ray[default]>=2.5.0 

ENV PYTHONPATH=/workspace
ENV HF_HOME=/workspace/cache
ENV TRANSFORMERS_CACHE=/workspace/cache

COPY docker/entrypoint.sh /workspace/docker/entrypoint.sh
RUN chmod +x /workspace/docker/entrypoint.sh

ENTRYPOINT ["/bin/bash", "/workspace/docker/entrypoint.sh"]
