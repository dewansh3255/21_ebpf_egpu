# Group 21 — GRS Project Part A
# Profiling CPU, Network Stack, and GPU Overheads in Containerized vs Non-Containerized ML Workloads using eBPF and eGPU

## Project Overview
This project profiles the performance overhead introduced by containerization (Docker with namespaces and cgroups) on ML training workloads running on multi-GPU servers. We use eBPF for CPU/kernel/network profiling and nvidia-smi based GPU monitoring (with eGPU integration planned for Part B).

## Team — Group 21
- Dewansh Khandelwal
- Palak Mishra
- Sanskar Goyal
- Yash Nimkar
- Kunal Verma

## Prerequisites

### System Requirements
- Linux (Ubuntu 20.04+ or similar)
- NVIDIA GPU(s) with CUDA 12.x drivers
- Docker with NVIDIA Container Toolkit
- Python 3.8+
- Root access (required for eBPF)

### Installation
```bash
# Install BCC (eBPF tooling)
sudo apt-get update
sudo apt-get install -y bpfcc-tools python3-bpfcc libbpfcc-dev

# Install Python dependencies
pip3 install torch torchvision matplotlib numpy psutil

# Install Docker (if not installed)
# Follow: https://docs.docker.com/engine/install/ubuntu/

# Install NVIDIA Container Toolkit
# Follow: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

## Project Structure
```
21_partA_ebpf_gpu_profiling/
├── README.md                       # This file
├── 21_cpu_profiler.py              # eBPF CPU scheduling profiler
├── 21_syscall_counter.py           # eBPF syscall counter + latency
├── 21_net_profiler.py              # eBPF network stack profiler
├── 21_ml_workload.py               # PyTorch DDP ResNet-18 on CIFAR-10
├── 21_gpu_monitor.py               # GPU monitoring (nvidia-smi based)
├── 21_container_setup.sh           # Docker container setup
├── 21_run_native.sh                # Run full pipeline (native)
├── 21_run_container.sh             # Run full pipeline (containerized)

```

## Usage

### Quick Start

**1. Build the Docker image:**
```bash
chmod +x 21_container_setup.sh
./21_container_setup.sh build
```

**2. Run native profiling:**
```bash
sudo ./21_run_native.sh --gpus 1 --epochs 5
```

**3. Run containerized profiling:**
```bash
sudo ./21_run_container.sh --gpus 1 --epochs 5
```

### Individual Tools

**CPU Profiler:**
```bash
sudo python3 21_cpu_profiler.py --duration 60 --output 21_cpu_results.csv
```

**Syscall Counter:**
```bash
sudo python3 21_syscall_counter.py --duration 60 --output 21_syscall_results.csv
```

**Network Profiler:**
```bash
sudo python3 21_net_profiler.py --duration 60 --output 21_net_results.csv
```

**GPU Monitor:**
```bash
python3 21_gpu_monitor.py --duration 60 --interval 0.5 --output 21_gpu_results.csv
```

**ML Workload (single GPU):**
```bash
python3 21_ml_workload.py --gpus 1 --epochs 5 --batch-size 128
```

**ML Workload (multi-GPU with DDP):**
```bash
torchrun --nproc_per_node=2 21_ml_workload.py --gpus 2 --epochs 5
```

## Methodology
1. Start eBPF profilers on the host to monitor CPU scheduling, syscalls, and network stack
2. Start GPU monitor to track GPU utilization
3. Run ML workload (ResNet-18 on CIFAR-10) either natively or in Docker container
4. Collect profiling data from all tools
5. Compare metrics between native and containerized runs

## References
- Yang et al., "eGPU: Extending eBPF Programmability and Observability to GPUs," HCDS '25
- BCC (BPF Compiler Collection): https://github.com/iovisor/bcc
- PyTorch DDP: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
