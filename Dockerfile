# Group 21 - Containerized ML Workload Environment
# Base: NVIDIA CUDA with cuDNN for GPU support
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    net-tools \
    iproute2 \
    linux-headers-generic \
    bpfcc-tools \
    python3-bpfcc \
    libbpfcc-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir matplotlib numpy psutil

# Set working directory
WORKDIR /workspace

# Copy project files
COPY G_21_*.py /workspace/

# Default command
CMD ["python3", "G_21_ml_workload.py", "--gpus", "1", "--epochs", "5"]
