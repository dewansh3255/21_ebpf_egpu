# Group 21 — GRS Project Part B (Final Submission)
# eBPF + eGPU: Profiling CPU, Network, GPU Overheads in Containerized vs Native ML Workloads

## Project Overview

This project profiles the performance overhead introduced by containerization (Docker) on ML training workloads running on multi-GPU servers. We use:

- **eBPF** for CPU scheduling, syscall, and TCP network profiling (kernel-level, zero-overhead tracing)
- **eGPU-inspired GPU tracing** via eBPF uprobes on `libcuda.so` for GPU kernel launch and memory transfer monitoring
- **nvidia-smi polling** for GPU utilization, power, and temperature monitoring

> **Note on eGPU**: The real eGPU framework (Yang et al., HCDS '25) JIT-compiles eBPF bytecode into NVIDIA PTX and injects probes directly into GPU kernels. Building the full eGPU framework requires LLVM ≥15, custom bpftime userspace runtime, and Frida-based binary rewriting — which could not be compiled in our environment (host LLVM 14, Docker image LLVM 10). We adopt a **hybrid approach**: using standard eBPF uprobes on the CUDA driver API (`libcuda.so`) to trace GPU operations from the CPU side, combined with nvidia-smi polling. This captures the same API-level events (kernel launches, memory transfers) without PTX-level instrumentation.

### What We Measure

| Profiling Layer | Tool | Technique |
|---|---|---|
| CPU Scheduling | `G_21_cpu_profiler.py` | eBPF tracepoints (`sched:sched_switch`) |
| System Calls | `G_21_syscall_counter.py` | eBPF tracepoints (`raw_syscalls:sys_enter/sys_exit`) |
| Network Stack | `G_21_net_profiler.py` | eBPF kprobes (`tcp_sendmsg`, `tcp_recvmsg`) |
| GPU Utilization | `G_21_gpu_monitor_nvidia.py` | nvidia-smi polling (util, mem, power, temp) |
| GPU Kernel Tracing | `G_21_egpu_monitor.py` | eBPF uprobes on `libcuda.so` (`cuLaunchKernel`, `cuMemcpy*`) — hybrid approach |
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
│── eGPU-Inspired GPU Profiler (eBPF uprobes) ───────────
├── G_21_egpu_monitor.py                # eBPF uprobes on libcuda.so (cuLaunchKernel, cuMemcpy*) — hybrid approach
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
    ├── plots_hardcoded/                # 11 hardcoded PNGs (from G_21_plot_hardcoded.py)
    └── system_info.json                # System specs
```

## Key Results

| Metric | Native | Container | Overhead |
|---|---|---|---|
| Training Time (sec) | 31.8 | 34.3 | **+7.9%** |
| Throughput (samples/sec) | 9,274 | 8,472 | **-8.6%** |
| Final Test Accuracy (%) | 81.1 | 81.1 | +0.0% |
| GPU 0 Avg Util (active, %) | 76.7 | 74.1 | -3.4% |
| GPU 0 Avg Power (W) | 158.4 | 198.9 | **+25.6%** |
| Sched Latency Mean (μs) | 13.9 | 17.7 | +27.3% |
| Sched Latency P95 (μs) | 15.2 | 14.4 | -5.3% |
| Total Syscalls | 11,926,534 | 8,180,511 | -31.4%† |
| Unique Syscall Types | 121 | 167 | **+38.0%** |
| TCP Send Avg Latency (μs) | 30.4 | 35.7 | +17.4% |

† Different profiling windows (native 61.1s vs container 43.6s); syscall rate is comparable.

**Key Findings:**
- Container adds **~7.9% training time overhead** and **-8.6% throughput** on H100 NVL GPUs
- GPU utilization during active training is near-identical (76.7% vs 74.1%) — GPU remains the bottleneck
- GPU power consumption is **+25.6% higher** in container, likely due to Docker namespace initialization
- Syscall diversity increases significantly (**+38% unique types**) due to container overlay FS and namespace management
- CPU scheduling latency mean is slightly higher in container (+27.3%) but P95/P99 are comparable
- Both configurations achieve identical final test accuracy (81.1%), confirming correctness

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

> **Note**: eBPF profilers (CPU, syscall, network) require system Python with PYTHONPATH for BCC.
> The GPU monitor uses the venv Python.

```bash
# eBPF CPU profiler (must use system python3 + PYTHONPATH)
sudo env PYTHONPATH=/usr/lib/python3/dist-packages python3 G_21_cpu_profiler.py --duration 60 --output cpu_results.csv

# eBPF Syscall counter
sudo env PYTHONPATH=/usr/lib/python3/dist-packages python3 G_21_syscall_counter.py --duration 60 --output syscall_results.csv

# eBPF Network profiler
sudo env PYTHONPATH=/usr/lib/python3/dist-packages python3 G_21_net_profiler.py --duration 60 --output net_results.csv

# GPU monitor (nvidia-smi, uses venv Python)
venv/bin/python3 G_21_gpu_monitor_nvidia.py --duration 60 --interval 0.1 --output gpu_results.csv

# eGPU monitor (eBPF uprobes on libcuda.so)
sudo env PYTHONPATH=/usr/lib/python3/dist-packages python3 G_21_egpu_monitor.py
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

### eGPU Monitor (`G_21_egpu_monitor.py`) — Hybrid Approach
- **Technique**: eBPF uprobes on `libcuda.so` functions (`cuLaunchKernel`, `cuMemcpyHtoD_v2`, `cuMemcpyDtoH_v2`)
- **Metrics**: GPU kernel launch events (COMPUTE_MATH) and memory transfer events (MEM_TRANSFER)
- **Output**: CSV trace with timestamp, event type, duration
- **Reference**: Inspired by eGPU (Yang et al., "eGPU: Extending eBPF Programmability to GPUs," HCDS '25)
- **Limitation**: Traces CUDA driver API calls from CPU side via uprobes (not PTX-level GPU kernel instrumentation). The full eGPU framework requires LLVM ≥15 and custom bpftime runtime which could not be built in our environment.

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

## Limitations

1. **eGPU Build**: The real eGPU framework (PTX-level GPU instrumentation) could not be built due to LLVM version constraints (requires ≥15, host has 14, Docker image has 10). We use a hybrid eBPF uprobe approach instead.
2. **eBPF Root Access**: All eBPF profilers require root/sudo privileges, limiting deployment in locked-down environments.
3. **Kernel Version**: BCC 0.18 on kernel 5.15 — some newer eBPF features (e.g., BPF LSM, ringbuf) are not available.
4. **Single-Node Only**: Experiments run on one server with 2 GPUs. Multi-node distributed training overhead is not captured.
5. **nvidia-smi Polling**: GPU metrics are sampled at 100ms intervals — sub-millisecond GPU events may be missed.

## References

- Yang et al., "eGPU: Extending eBPF Programmability and Observability to GPUs," HCDS '25
- BCC (BPF Compiler Collection): https://github.com/iovisor/bcc
- PyTorch DDP: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/

## Repository

- GitHub: https://github.com/dewansh3255/21_ebpf_egpu
- Partner: https://github.com/Sanskargoyal608/eBPF-eGPU
