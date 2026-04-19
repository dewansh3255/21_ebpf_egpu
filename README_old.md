# Group 21 — GRS Project Part A
# Profiling CPU, Network Stack, and GPU Overheads in Containerized vs Non-Containerized ML Workloads using eBPF

## Project Overview
This project profiles the performance overhead introduced by containerization (Docker with namespaces and cgroups) on ML training workloads running on multi-GPU servers. We use **eBPF** for CPU/kernel/network profiling and **GPU monitoring** (currently nvidia-smi based, with eGPU integration planned) to capture comprehensive system-level metrics.

**Implementation Status:**

✅ **Configuration A (main branch)**: Single-server multi-GPU profiling  
✅ **Configuration B (objective-2-tcp branch)**: Two-laptop distributed TCP profiling  

### Configuration A: Single-Server Multi-GPU (Current Branch)
Compares **native (bare-metal)** vs **containerized (Docker)** execution of PyTorch DDP ResNet-18 training on CIFAR-10, measuring:
- **CPU scheduling latency** and context switches (via `sched_switch` tracepoints)
- **System call frequency** and latency distribution (via `sys_enter`/`sys_exit` tracepoints)
- **Network stack overhead** for DDP communication (via TCP kprobes on `tcp_sendmsg`/`tcp_recvmsg`)
- **GPU utilization**, memory usage, and power consumption (via nvidia-smi polling)

### Configuration B: Distributed TCP Training (objective-2-tcp branch)
Two-laptop federated learning setup with eBPF network profiling:
- **Federated learning** over TCP/IP with FastAPI-based server-client architecture
- **Network stack profiling** with eBPF kprobes on TCP send/receive paths (port 8000 filtering)
- **GPU monitoring** with eBPF tracing of GPU IOCTL calls
- **ResNet-18 on CIFAR-10** for gradient aggregation over TCP
- Real-world distributed training scenario with network compression (zlib)

**Model Choice:** ResNet-18 selected due to GPU memory constraints on distributed laptops (ResNet-50 originally planned but not feasible on available hardware).

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

### Main Branch (Configuration A)
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

### objective-2-tcp Branch (Configuration B)
```
21_ebpf_egpu/
├── src/
│   ├── client.py                   # Federated learning client (runs on each laptop)
│   ├── server.py                   # FastAPI-based aggregation server
│   ├── main.py                     # Launcher script
│   └── dataset.py                  # CIFAR-10 data partitioning
│
└── monitor/
    ├── monitor_network.py          # eBPF TCP send/recv profiler (port 8000)
    ├── monitor_network_k.py        # Enhanced network monitoring
    ├── monitor_gpu.py              # eBPF GPU IOCTL tracing
    ├── analyze_trace.py            # Post-processing analysis
    └── loader.py                   # Data loading utilities
```

## Usage

### Switching Between Configurations

```bash
# For single-server containerization profiling (Config A)
git checkout main

# For distributed TCP training profiling (Config B)
git checkout objective-2-tcp
```

### Configuration A: Single-Server Multi-GPU (main branch)

**Quick Start:**

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

### Configuration B: Distributed TCP Training (objective-2-tcp branch)

**Setup:**

```bash
# Switch to the distributed TCP branch
git checkout objective-2-tcp

# On Server Laptop (e.g., 192.168.52.110)
cd src
python3 server.py

# On Client Laptop
cd src
python3 client.py

# In separate terminals, run eBPF monitors:
# Terminal 1: Network monitoring
sudo python3 monitor/monitor_network.py

# Terminal 2: GPU monitoring
sudo python3 monitor/monitor_gpu.py

# Analyze results
python3 monitor/analyze_trace.py
```

**Architecture:**
- **Federated Learning**: Clients train locally, send gradients to server for aggregation
- **Network Profiling**: eBPF traces on `tcp_sendmsg`/`tcp_recvmsg` (port 8000)
- **Compression**: zlib-based gradient compression to reduce TCP overhead
- **Synchronization**: Round-based barrier synchronization via REST API

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

### Configuration A: Single-Server Multi-GPU Profiling (main branch)

The profiling pipeline follows these steps:

1. **Start eBPF profilers** on the host system:
   - **CPU scheduler** tracepoints (`sched:sched_switch`) — context switches, scheduling delays
   - **Syscall interface** hooks (`raw_syscalls:sys_enter`, `raw_syscalls:sys_exit`) — syscall frequency and latency
   - **Network stack** kprobes (`tcp_sendmsg`, `tcp_recvmsg`) — TCP send/receive overhead for DDP AllReduce

2. **Start GPU monitor** polling nvidia-smi at 0.5s intervals for utilization, memory, and power metrics

3. **Run ML workload** (ResNet-18 training on CIFAR-10 with PyTorch DDP):
   - **Native**: Direct execution on host (bare-metal)
   - **Containerized**: Execution inside Docker with `--gpus all`, namespace isolation, and cgroup resource controls

4. **Collect profiling data** from all tools in timestamped CSV/JSON format

5. **Compare metrics** between native and containerized runs:
   - Training time (wall clock) and per-epoch duration
   - CPU scheduling overhead and context switch frequency
   - System call patterns and latency distribution
   - Network communication overhead for multi-GPU DDP (AllReduce via NCCL over PCIe)
   - GPU utilization and power consumption

### Configuration B: Distributed TCP Training Profiling (objective-2-tcp branch)

Federated learning architecture with network-level profiling:

1. **Server setup** (Laptop 1):
   - FastAPI-based aggregation server receives gradient updates from clients
   - Maintains global ResNet-18 model
   - Performs FedAvg aggregation after each round
   - Evaluates global model on CIFAR-10 test set

2. **Client setup** (Laptop 2):
   - Trains local ResNet-18 replica on data partition (5000 images)
   - Sends compressed gradients to server via HTTP POST (zlib compression)
   - Receives updated global weights after aggregation
   - 3 local epochs per round, 10 global rounds

3. **eBPF network profiling**:
   - `monitor_network.py`: Traces TCP send/recv on port 8000 (server port)
   - Captures: timestamp, direction (send/recv), payload size, IP addresses, PID/TID
   - CSV output with microsecond timestamps for latency analysis

4. **eBPF GPU profiling**:
   - `monitor_gpu.py`: Traces GPU IOCTL system calls
   - Correlates GPU activity with network events
   - Identifies GPU idle time during gradient transmission

**Alignment with Proposal:**
- ✅ **Objective 1**: Containerization overhead quantification (Config A) — **Fully Implemented**
- ✅ **Objective 2**: TCP network stack profiling (Config B, two-laptop setup) — **Fully Implemented**
- ✅ **Objective 3**: Cross-scenario comparison — **Data collected, analysis in progress**
- ⏳ **eGPU Integration**: GPU-side PTX injection for kernel-level instrumentation — **Planned for future work**

**Model Selection Rationale:**
ResNet-18 selected instead of ResNet-50 due to GPU memory constraints on laptops used for distributed training (Configuration B). ResNet-18 provides sufficient complexity for profiling containerization and network overhead while fitting within available GPU memory (typical laptop GPUs: 4-6GB VRAM).

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
✅ **Multi-GPU support** — PyTorch DDP (DistributedDataParallel) training on single-server multi-GPU setup  
✅ **Distributed TCP profiling** — Two-laptop federated learning with network stack instrumentation  
✅ **Host-side monitoring** — Profilers run on host to capture container overhead accurately  
✅ **Comprehensive metrics** — CPU, syscalls, network stack (TCP send/recv), and GPU utilization  
✅ **Automated pipeline** — Single-command execution for reproducible experiments (Config A)  
✅ **Network compression** — zlib-based gradient compression for realistic distributed training (Config B)  
✅ **Visualization** — Auto-generated plots for comparative analysis  
✅ **Two configurations** — Single-server containerization study + distributed TCP overhead study  

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

## Project Roadmap

### ✅ Configuration A (main branch) — Completed
- [x] Single-server multi-GPU profiling infrastructure
- [x] eBPF-based CPU, syscall, and network stack profilers
- [x] Native vs containerized comparison methodology
- [x] ResNet-18 DDP workload on CIFAR-10
- [x] Automated visualization pipeline
- [x] Docker containerization with GPU passthrough
- [x] Comprehensive metrics collection (CPU, syscalls, network, GPU)

### ✅ Configuration B (objective-2-tcp branch) — Completed
- [x] Two-laptop distributed federated learning setup
- [x] FastAPI-based server-client architecture
- [x] eBPF TCP network profiling (port 8000 filtering)
- [x] eBPF GPU IOCTL tracing
- [x] Network compression (zlib) for gradient transmission
- [x] Round-based synchronization mechanism
- [x] ResNet-18 on CIFAR-10 with data partitioning

### 🔄 Phase 3 (Future Work) — Planned
- [ ] eGPU integration for GPU kernel instrumentation via PTX injection
- [ ] GPU-kernel timeline correlation with CPU-side eBPF events
- [ ] Unified cross-scenario comparison dashboard (Objectives 1+2+3)
- [ ] Extended models if more powerful GPUs become available (ResNet-50, BERT-base)
- [ ] Performance optimization recommendations based on profiling data

### 📊 Current Analysis Status
- [x] Configuration A: Data collected, visualizations generated
- [x] Configuration B: Data collected, network traces captured
- [ ] Cross-configuration comparative analysis (in progress)
- [ ] Final report with recommendations (in progress)

---

**Group 21 — Graduate Research Seminar (GRS) Project**  
*Profiling CPU, Network Stack, and GPU Overheads in Containerized ML Workloads*
