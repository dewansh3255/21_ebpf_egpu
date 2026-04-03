# Group 21 — GRS Project Part A
# Profiling CPU, Network Stack, and GPU Overheads in Containerized vs Non-Containerized ML Workloads using eBPF

## Project Overview
This project profiles the performance overhead introduced by containerization (Docker with namespaces and cgroups) on ML training workloads running on multi-GPU servers. We use **eBPF** for CPU/kernel/network profiling and **nvidia-smi based GPU monitoring** to capture comprehensive system-level metrics.

The project compares **native (bare-metal)** vs **containerized (Docker)** execution of a PyTorch DDP ResNet-18 training workload on CIFAR-10, measuring:
- **CPU scheduling latency** and context switches
- **System call frequency** and latency distribution
- **Network stack overhead** for DDP communication
- **GPU utilization**, memory usage, and power consumption

## Team — Group 21
- Dewansh Khandelwal
- Palak Mishra
- Sanskar Goyal
- Yash Nimkar
- Kunal Verma

## Prerequisites

### System Requirements
- Linux (Ubuntu 20.04+ or 22.04 recommended)
- NVIDIA GPU(s) with CUDA 12.1+ drivers
- Docker with NVIDIA Container Toolkit (nvidia-docker2)
- Python 3.8+
- Root/sudo access (required for eBPF profilers)
- BCC (BPF Compiler Collection) tools

### Installation

**1. Install BCC (eBPF tooling):**
```bash
sudo apt-get update
sudo apt-get install -y bpfcc-tools python3-bpfcc libbpfcc-dev linux-headers-$(uname -r)
```

**2. Install Python dependencies:**
```bash
pip3 install torch torchvision matplotlib numpy psutil pandas
```

**3. Install Docker (if not installed):**
```bash
# Follow official guide: https://docs.docker.com/engine/install/ubuntu/
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

**4. Install NVIDIA Container Toolkit:**
```bash
# Follow: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## Project Structure
```
21_ebpf_egpu/
├── README.md                       # This file
├── Dockerfile                      # Container image with CUDA, PyTorch, and eBPF tools
│
├── 21_cpu_profiler.py              # eBPF CPU scheduling profiler
├── 21_syscall_counter.py           # eBPF syscall counter with latency tracking
├── 21_net_profiler.py              # eBPF network stack profiler
├── 21_gpu_monitor.py               # GPU monitoring (nvidia-smi based)
├── 21_ml_workload.py               # PyTorch DDP ResNet-18 on CIFAR-10
├── 21_plot_results.py              # Visualization script for results
│
├── 21_container_setup.sh           # Docker image builder
├── 21_run_native.sh                # Run full profiling pipeline (native)
├── 21_run_container.sh             # Run full profiling pipeline (containerized)
│
└── results/                        # Output directory
    ├── native/                     # Native run results (CSV + JSON)
    ├── container/                  # Container run results (CSV + JSON)
    ├── native_*.png                # Visualization plots for native
    └── container_*.png             # Visualization plots for container
```

## Usage

### Quick Start (Recommended)

**1. Build the Docker image:**
```bash
chmod +x 21_container_setup.sh
./21_container_setup.sh build
```

**2. Run native profiling:**
```bash
sudo ./21_run_native.sh --gpus 1 --epochs 5 --duration 120
```

**3. Run containerized profiling:**
```bash
sudo ./21_run_container.sh --gpus 1 --epochs 5 --duration 120
```

**4. Generate comparison plots:**
```bash
python3 21_plot_results.py
```

This will generate visualization plots in the `results/` directory:
- `native_gpu_timeline.png` / `container_gpu_timeline.png` — GPU utilization and power over time
- `native_syscall_counts.png` / `container_syscall_counts.png` — Top system calls by frequency
- `native_net_latency.png` / `container_net_latency.png` — Network latency distribution

### Individual Tools

Run profilers separately for custom monitoring:

**CPU Profiler:**
```bash
sudo python3 21_cpu_profiler.py --duration 60 --output cpu_results.csv
```
Captures context switches, scheduling latency, and per-process CPU time.

**Syscall Counter:**
```bash
sudo python3 21_syscall_counter.py --duration 60 --output syscall_results.csv
```
Tracks system call frequency and latency distribution.

**Network Profiler:**
```bash
sudo python3 21_net_profiler.py --duration 60 --output net_results.csv
```
Monitors network stack overhead (packet send/receive latency).

**GPU Monitor:**
```bash
python3 21_gpu_monitor.py --duration 60 --interval 0.5 --output gpu_results.csv
```
Polls GPU utilization, memory usage, power draw via nvidia-smi.

**ML Workload (single GPU):**
```bash
python3 21_ml_workload.py --gpus 1 --epochs 5 --batch-size 128
```

**ML Workload (multi-GPU with DDP):**
```bash
torchrun --nproc_per_node=2 21_ml_workload.py --gpus 2 --epochs 5
```

## Methodology

The profiling pipeline follows these steps:

1. **Start eBPF profilers** on the host system:
   - CPU scheduler tracepoints (`sched:sched_switch`, `sched:sched_wakeup`)
   - Syscall entry/exit hooks (`raw_syscalls:sys_enter`, `raw_syscalls:sys_exit`)
   - Network stack tracepoints (`net:net_dev_start_xmit`, `net:netif_receive_skb`)

2. **Start GPU monitor** polling nvidia-smi at 0.5s intervals

3. **Run ML workload** (ResNet-18 training on CIFAR-10):
   - Native: Direct execution on host
   - Container: Execution inside Docker with `--gpus all` and namespace isolation

4. **Collect profiling data** from all tools (CSV + JSON format)

5. **Compare metrics** between native and containerized runs:
   - Training time (wall clock)
   - CPU scheduling overhead
   - System call patterns and latency
   - Network communication overhead (for multi-GPU DDP)
   - GPU utilization and power consumption

## Output Files

After running the profiling scripts, results are saved in `results/native/` and `results/container/`:

| File | Description |
|------|-------------|
| `21_cpu_results.csv` | Context switch count, scheduling latency per process |
| `21_syscall_results.csv` | Syscall name, count, min/avg/max latency |
| `21_net_results.csv` | Network send/receive latency per packet |
| `21_gpu_results.csv` | GPU utilization, memory usage, power draw (time series) |
| `21_training_*.json` | Training metrics (epoch times, loss, accuracy) |
| `21_container_inspect.json` | Docker container metadata (containerized only) |

Visualization plots (generated by `21_plot_results.py`):
- `native_gpu_timeline.png` / `container_gpu_timeline.png`
- `native_syscall_counts.png` / `container_syscall_counts.png`
- `native_net_latency.png` / `container_net_latency.png`

## Key Features

✅ **eBPF-based profiling** — Zero-overhead kernel-level tracing without modifying application code  
✅ **Multi-GPU support** — PyTorch DDP (DistributedDataParallel) training with 1-4 GPUs  
✅ **Host-side monitoring** — Profilers run on host to capture container overhead accurately  
✅ **Comprehensive metrics** — CPU, syscalls, network stack, and GPU utilization  
✅ **Automated pipeline** — Single-command execution for reproducible experiments  
✅ **Visualization** — Auto-generated plots for comparative analysis  

## Troubleshooting

**"ERROR: BCC (BPF Compiler Collection) is not installed"**
```bash
sudo apt-get install bpfcc-tools python3-bpfcc libbpfcc-dev
```

**"ERROR: eBPF profilers require root privileges"**
```bash
# Run with sudo
sudo ./21_run_native.sh
```

**"docker: Error response from daemon: could not select device driver"**
```bash
# Install NVIDIA Container Toolkit
sudo apt-get install nvidia-docker2
sudo systemctl restart docker
```

**"CUDA out of memory"**
```bash
# Reduce batch size
python3 21_ml_workload.py --gpus 1 --batch-size 64
```

**Profilers not capturing container processes**
- Ensure profilers run on **host** (not inside container)
- The scripts `21_run_container.sh` already handles this correctly

## References
- Yang et al., "eGPU: Extending eBPF Programmability and Observability to GPUs," HCDS '25
- BCC (BPF Compiler Collection): https://github.com/iovisor/bcc
- PyTorch DDP: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/

---

**Group 21 — Graduate Research Seminar (GRS) Project**  
*Profiling Containerized ML Workloads with eBPF*
