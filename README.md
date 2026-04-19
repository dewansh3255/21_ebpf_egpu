# Group 21 — GRS Project Part B (Final Submission)
# eBPF + eGPU: Profiling CPU, Network, GPU Overheads in Containerized vs Native ML Workloads

## Project Overview

This project profiles the performance overhead introduced by containerization (Docker) on ML training workloads running on multi-GPU servers. We use:

- **eBPF** for CPU scheduling, syscall, and TCP network profiling (kernel-level, zero-overhead tracing)
- **eGPU** (eBPF for GPUs) for GPU kernel launch and memory transfer tracing via `libcuda.so` uprobes
- **nvidia-smi polling** for GPU utilization, power, and temperature monitoring

### What We Measure

| Profiling Layer | Tool | Technique |
|---|---|---|
| CPU Scheduling | `G_21_cpu_profiler.py` | eBPF tracepoints (`sched:sched_switch`) |
| System Calls | `G_21_syscall_counter.py` | eBPF tracepoints (`raw_syscalls:sys_enter/sys_exit`) |
| Network Stack | `G_21_net_profiler.py` | eBPF kprobes (`tcp_sendmsg`, `tcp_recvmsg`) |
| GPU Utilization | `G_21_gpu_monitor_nvidia.py` | nvidia-smi polling (util, mem, power, temp) |
| GPU Kernel Tracing | `G_21_egpu_monitor.py` | eBPF uprobes on `libcuda.so` (`cuLaunchKernel`, `cuMemcpy*`) |
| Network (FL) | `G_21_egpu_net_monitor.py` | eBPF kprobes + tracepoints (TCP + scheduler) |

### Experiment Setup

- **Hardware**: 2x NVIDIA H100 NVL (95830 MiB each), AMD EPYC 9354 32-Core, 503.6 GB RAM
- **Software**: Ubuntu 22.04, Kernel 5.15.0-161-generic, CUDA 12.1.1, PyTorch 2.5.1+cu121, Docker 29.3.1
- **Workload**: ResNet-18 on CIFAR-10, 10 epochs, batch_size=128, PyTorch DDP across 2 GPUs
- **Methodology**: 3 trials each for native and container, median trial selected

## Team — Group 21

- Dewansh Khandelwal
- Palak Mishra
- Sanskar Goyal
- Yash Nimkar
- Kunal Verma

## Project Structure

```
G_21_Part_B_eBPF_eGPU/
│
├── README.md                           # This file
├── Dockerfile                          # CUDA 12.1 + PyTorch + BCC container image
│
│── eBPF Profilers (CPU/Syscall/Network) ────────────────
├── G_21_cpu_profiler.py                # eBPF CPU scheduler profiler (ctx switches + sched latency)
├── G_21_syscall_counter.py             # eBPF syscall counter with per-call latency
├── G_21_net_profiler.py                # eBPF TCP send/recv network profiler
│
│── eGPU Profiler (GPU via eBPF) ────────────────────────
├── G_21_egpu_monitor.py                # eBPF uprobes on libcuda.so (cuLaunchKernel, cuMemcpy*)
├── G_21_egpu_net_monitor.py            # eBPF network + scheduler monitor (FL scenario)
│
│── GPU Monitor (nvidia-smi) ────────────────────────────
├── G_21_gpu_monitor_nvidia.py          # nvidia-smi based GPU polling (util, mem, power, temp)
│
│── ML Workload ─────────────────────────────────────────
├── G_21_ml_workload.py                 # PyTorch DDP ResNet-18 on CIFAR-10 (multi-GPU)
│
│── Federated Learning (Partner) ────────────────────────
├── G_21_federated_server.py            # FastAPI federated learning server (FedAvg)
├── G_21_federated_client.py            # FL client (ResNet-18, gradient upload)
├── G_21_federated_dataset.py           # CIFAR-10 data partitioning for FL
├── G_21_federated_main.py              # Local FL simulation (2 clients, 3 rounds)
│
│── Orchestration Scripts ───────────────────────────────
├── G_21_run_experiment.sh              # Master experiment runner (3 trials each)
├── G_21_run_native.sh                  # Native profiling orchestrator
├── G_21_run_container.sh               # Container profiling orchestrator
├── G_21_container_setup.sh             # Docker image builder & container manager
├── G_21_local_dry_run.sh               # FL dry run with eBPF monitoring
│
│── Analysis & Visualization ────────────────────────────
├── G_21_compare_results.py             # Generate 11 comparison plots + JSON summaries
├── G_21_analyze_cpu.py                 # CPU CSV analyzer with comparison mode
├── G_21_plot_results.py                # Per-scenario plotting
├── G_21_plot_hardcoded.py              # Hardcoded matplotlib plots (no CSV input)
├── G_21_unified_timeline.py            # 3-row dashboard (GPU + CPU + Network)
│
│── Data & Results ──────────────────────────────────────
├── data/cifar-10-batches-py/           # CIFAR-10 dataset
└── results/
    ├── native/                         # Native run CSVs + training JSON
    ├── container/                      # Container run CSVs + training JSON + inspect JSON
    ├── plots/                          # 11 comparison PNGs + 2 analysis JSONs
    ├── plots_hardcoded/                # 11 hardcoded PNGs (from G_21_plot_hardcoded.py)
    └── system_info.json                # System specs
```

## Key Results (Hardcoded Summary)

| Metric | Native | Container | Overhead |
|---|---|---|---|
| Training Time (sec) | 118.3 | 121.3 | **+2.5%** |
| Throughput (samples/sec) | 2,368 | 2,301 | -2.8% |
| GPU Avg Utilization (%) | 99.9 | 99.9 | +0.0% |
| GPU Avg Power (W) | 213.8 | 220.1 | +2.9% |
| Sched Latency Mean (us) | 19.71 | 18.07 | -8.3% |
| Sched Latency P95 (us) | 12.27 | 12.37 | +0.8% |
| Total Syscalls | 9,521,613 | 9,766,704 | +2.6% |
| Unique Syscall Types | 122 | 172 | +41.0% |
| TCP Send Avg Latency (us) | 38.48 | 34.35 | -10.7% |
| TCP Recv Avg Latency (us) | 5.95 | 6.72 | +12.8% |

**Key Findings:**
- Container overhead is minimal (~2.5% training time) on H100 NVL GPUs
- GPU utilization is near-identical (99.9% both), GPU is the bottleneck, not containers
- Syscall diversity increases (+41% unique types) due to container namespace management
- CPU scheduling latency is comparable (P95 within 1%)
- Network overhead varies: TCP send is faster in containers, TCP recv is slower

## Prerequisites

### System Requirements
- Linux (Ubuntu 20.04+ or 22.04)
- NVIDIA GPU(s) with CUDA 12.1+ drivers
- Docker with NVIDIA Container Toolkit
- Python 3.8+
- Root/sudo access (required for eBPF profilers)
- BCC (BPF Compiler Collection) tools

### Installation

```bash
# 1. Install BCC (eBPF tooling)
sudo apt-get update
sudo apt-get install -y bpfcc-tools python3-bpfcc libbpfcc-dev linux-headers-$(uname -r)

# 2. Install Python dependencies
pip3 install torch torchvision matplotlib numpy psutil pandas seaborn fastapi uvicorn requests

# 3. Install Docker + NVIDIA Container Toolkit
curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh
# Then follow: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

## Usage

### Run Full Experiment (3 trials native + 3 trials container)

```bash
sudo ./G_21_run_experiment.sh --gpus 2 --epochs 10
```

### Run Individual Components

```bash
# Native profiling only
sudo ./G_21_run_native.sh --gpus 2 --epochs 10 --duration 180

# Container profiling only
sudo ./G_21_run_container.sh --gpus 2 --epochs 10 --duration 180

# Build Docker image
./G_21_container_setup.sh build
```

### Run Individual Profilers

```bash
# eBPF CPU profiler
sudo python3 G_21_cpu_profiler.py --duration 60 --output cpu_results.csv

# eBPF Syscall counter
sudo python3 G_21_syscall_counter.py --duration 60 --output syscall_results.csv

# eBPF Network profiler
sudo python3 G_21_net_profiler.py --duration 60 --output net_results.csv

# GPU monitor (nvidia-smi)
python3 G_21_gpu_monitor_nvidia.py --duration 60 --interval 0.1 --output gpu_results.csv

# eGPU monitor (eBPF uprobes on libcuda.so)
sudo python3 G_21_egpu_monitor.py
```

### Generate Plots

```bash
# Hardcoded plots (no CSV input required)
python3 G_21_plot_hardcoded.py

# Comparison plots from CSVs
python3 G_21_compare_results.py

# CPU analysis
python3 G_21_analyze_cpu.py --compare results/native results/container
```

### Federated Learning (Partner Component)

```bash
# Local dry run with eBPF monitoring
sudo ./G_21_local_dry_run.sh

# Or manually:
# Terminal 1: Start FL server
python3 G_21_federated_server.py

# Terminal 2: Start FL client
python3 G_21_federated_client.py

# Terminal 3: Start eGPU monitor
sudo python3 G_21_egpu_monitor.py

# Terminal 4: Start network monitor
sudo python3 G_21_egpu_net_monitor.py
```

## eBPF Profiling Details

### CPU Profiler (`G_21_cpu_profiler.py`)
- **Technique**: Attaches to `sched:sched_switch` tracepoint
- **Metrics**: Context switch count per process, scheduling latency (time on run queue)
- **Output**: Two CSVs — `ctx_switches.csv` and `sched_latency.csv`

### Syscall Counter (`G_21_syscall_counter.py`)
- **Technique**: Attaches to `raw_syscalls:sys_enter` and `raw_syscalls:sys_exit` tracepoints
- **Metrics**: Per-syscall count, total/avg/min/max latency
- **Output**: CSV with syscall number, name, count, latency statistics

### Network Profiler (`G_21_net_profiler.py`)
- **Technique**: kprobes on `tcp_sendmsg` and `tcp_recvmsg`
- **Metrics**: TCP send/recv latency per packet, byte counts, process info
- **Output**: CSV with timestamp, PID, comm, event_type, latency_ns, bytes

### eGPU Monitor (`G_21_egpu_monitor.py`)
- **Technique**: eBPF uprobes on `libcuda.so` functions (`cuLaunchKernel`, `cuMemcpyHtoD_v2`, `cuMemcpyDtoH_v2`)
- **Metrics**: GPU kernel launch events (COMPUTE_MATH) and memory transfer events (MEM_TRANSFER)
- **Output**: CSV trace with timestamp, event type, duration
- **Reference**: Based on eGPU paper (Yang et al., "eGPU: Extending eBPF Programmability to GPUs," HCDS '25)

## Methodology

1. **Start eBPF profilers** on host (CPU, syscall, network, eGPU)
2. **Start GPU monitor** (nvidia-smi polling at 100ms intervals)
3. **Run ML workload** (ResNet-18 DDP training):
   - **Native**: Direct execution on bare metal
   - **Container**: Docker with `--gpus all`, namespace isolation, cgroups
4. **Collect results** in timestamped CSV/JSON format
5. **Compare metrics** between native and containerized runs
6. **Generate visualizations** with hardcoded matplotlib plots

## Hardcoded Plots (G_21_plot_hardcoded.py)

All 11 plots generated with hardcoded values from experiment results:

1. **Training Loss** — Loss curve: Native vs Container
2. **Training Accuracy** — Train/test accuracy per epoch
3. **Throughput** — Samples/sec per epoch (bar chart)
4. **Epoch Time** — Per-epoch training time
5. **Total Comparison** — Training time, throughput, GPU util, power
6. **GPU Metrics** — Utilization, power, temperature over time
7. **Syscall Comparison** — Top 10 syscalls by count
8. **Syscall Latency** — Average latency per syscall type
9. **Network Comparison** — TCP send/recv latency and counts
10. **Scheduler Latency** — Mean/P95/P99 + context switches
11. **Overhead Summary Table** — All metrics in a single table

## References

- Yang et al., "eGPU: Extending eBPF Programmability and Observability to GPUs," HCDS '25
- BCC (BPF Compiler Collection): https://github.com/iovisor/bcc
- PyTorch DDP: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/

## Repository

- GitHub: https://github.com/dewansh3255/21_ebpf_egpu
- Partner: https://github.com/Sanskargoyal608/eBPF-eGPU
