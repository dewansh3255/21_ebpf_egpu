#!/bin/bash
# 21_container_setup.sh
# Group 21 - GRS Project Part A
#
# Sets up the Docker container environment for running ML workloads
# with eBPF profiling capabilities. Creates a container with GPU access,
# PyTorch, and BCC tools installed.
#
# Usage:
#   chmod +x 21_container_setup.sh
#   ./21_container_setup.sh build    # Build the Docker image
#   ./21_container_setup.sh run      # Run ML workload in container
#   ./21_container_setup.sh shell    # Get interactive shell in container
#
# Authors: Dewansh Khandelwal, Palak Mishra, Sanskar Goyal, Yash Nimkar, Kunal Verma

set -e

IMAGE_NAME="group21-ml-profiling"
CONTAINER_NAME="group21-ml-container"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- Dockerfile Creation ----
create_dockerfile() {
    cat > "${PROJECT_DIR}/Dockerfile" << 'DOCKERFILE_EOF'
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
RUN pip3 install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch \
    torchvision \
    matplotlib \
    numpy \
    psutil

# Set working directory
WORKDIR /workspace

# Copy project files
COPY 21_*.py /workspace/
COPY 21_*.sh /workspace/

# Default command
CMD ["python3", "21_ml_workload.py", "--gpus", "1", "--epochs", "5"]
DOCKERFILE_EOF
    echo "[Setup] Dockerfile created at ${PROJECT_DIR}/Dockerfile"
}

# ---- Build Docker Image ----
build_image() {
    echo "============================================"
    echo " Building Docker Image: ${IMAGE_NAME}"
    echo "============================================"
    create_dockerfile
    docker build -t "${IMAGE_NAME}" "${PROJECT_DIR}"
    echo "[Setup] Image built successfully: ${IMAGE_NAME}"
    echo "[Setup] Image size: $(docker image inspect ${IMAGE_NAME} --format='{{.Size}}' | numfmt --to=iec 2>/dev/null || docker image inspect ${IMAGE_NAME} --format='{{.Size}}')"
}

# ---- Run ML Workload in Container ----
run_workload() {
    local GPU_COUNT="${1:-1}"
    local EPOCHS="${2:-5}"

    echo "============================================"
    echo " Running ML Workload in Container"
    echo " GPUs: ${GPU_COUNT}, Epochs: ${EPOCHS}"
    echo "============================================"

    # Remove existing container if present
    docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

    if [ "${GPU_COUNT}" -eq 1 ]; then
        docker run \
            --name "${CONTAINER_NAME}" \
            --gpus all \
            --shm-size=2g \
            --ulimit memlock=-1 \
            -v "${PROJECT_DIR}/results:/workspace/results" \
            "${IMAGE_NAME}" \
            python3 21_ml_workload.py \
                --gpus 1 \
                --epochs "${EPOCHS}" \
                --output results/21_training_container.json
    else
        docker run \
            --name "${CONTAINER_NAME}" \
            --gpus all \
            --shm-size=2g \
            --ulimit memlock=-1 \
            --network=host \
            --ipc=host \
            -v "${PROJECT_DIR}/results:/workspace/results" \
            "${IMAGE_NAME}" \
            bash -c "torchrun --nproc_per_node=${GPU_COUNT} 21_ml_workload.py \
                --gpus ${GPU_COUNT} \
                --epochs ${EPOCHS} \
                --output results/21_training_container.json"
    fi

    echo "[Setup] Container workload complete."
}

# ---- Interactive Shell ----
run_shell() {
    echo "============================================"
    echo " Interactive Shell in Container"
    echo "============================================"

    docker rm -f "${CONTAINER_NAME}-shell" 2>/dev/null || true

    docker run -it \
        --name "${CONTAINER_NAME}-shell" \
        --gpus all \
        --shm-size=2g \
        --ulimit memlock=-1 \
        --network=host \
        --ipc=host \
        --privileged \
        -v "${PROJECT_DIR}:/workspace" \
        "${IMAGE_NAME}" \
        /bin/bash
}

# ---- Check Prerequisites ----
check_prereqs() {
    echo "[Setup] Checking prerequisites..."

    # Docker
    if ! command -v docker &> /dev/null; then
        echo "ERROR: Docker not found. Install Docker first."
        exit 1
    fi
    echo "  ✓ Docker found: $(docker --version)"

    # NVIDIA Container Toolkit
    if docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
        echo "  ✓ NVIDIA Container Toolkit working"
    else
        echo "  ✗ NVIDIA Container Toolkit not working"
        echo "    Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        echo "    Continuing anyway (GPU access in container may fail)..."
    fi

    # nvidia-smi on host
    if command -v nvidia-smi &> /dev/null; then
        echo "  ✓ nvidia-smi found"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        echo "  ✗ nvidia-smi not found on host"
    fi
}

# ---- Main ----
case "${1}" in
    build)
        check_prereqs
        build_image
        ;;
    run)
        run_workload "${2:-1}" "${3:-5}"
        ;;
    shell)
        run_shell
        ;;
    check)
        check_prereqs
        ;;
    *)
        echo "Usage: $0 {build|run|shell|check}"
        echo ""
        echo "  build          - Build the Docker image"
        echo "  run [gpus] [e] - Run workload with N GPUs for E epochs"
        echo "  shell          - Interactive shell in container"
        echo "  check          - Check prerequisites"
        exit 1
        ;;
esac
