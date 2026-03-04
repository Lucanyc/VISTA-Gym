FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set python3 as default
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /workspace

# Install PyTorch (separate due to custom index URL)
RUN pip3 install --no-cache-dir \
    torch==2.1.0+cu121 \
    torchvision==0.16.0+cu121 \
    triton==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install Python dependencies
COPY docker/requirement.txt /workspace/docker/requirement.txt
RUN pip3 install --no-cache-dir -r /workspace/docker/requirement.txt

ENV PYTHONPATH=/workspace
ENV HF_HOME=/workspace/cache
ENV TRANSFORMERS_CACHE=/workspace/cache

COPY docker/entrypoint.sh /workspace/docker/entrypoint.sh
RUN chmod +x /workspace/docker/entrypoint.sh

ENTRYPOINT ["/bin/bash", "/workspace/docker/entrypoint.sh"]
