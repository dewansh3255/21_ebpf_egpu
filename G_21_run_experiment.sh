#!/bin/bash
# 21_run_experiment.sh
# Group 21 - GRS Project Part A
#
# Master experiment runner with multi-trial support, GPU cooldown,
# and warmup epoch handling. Produces statistically rigorous results.
#
# Usage:
#   sudo ./21_run_experiment.sh [--trials N] [--gpus N] [--epochs E] [--duration D] [--cooldown C]
#
# Authors: Dewansh Khandelwal, Palak Mishra, Sanskar Goyal, Yash Nimkar, Kunal Verma

set -e

# ---- Cleanup trap: kill any orphaned profiler processes on exit ----
PROFILER_PIDS=""
cleanup() {
    echo ""
    echo ">>> Cleaning up background profilers..."
    for pid in ${PROFILER_PIDS}; do
        kill "${pid}" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    # Clean up any leftover containers
    docker rm -f "${CONTAINER_NAME:-group21-profiled-run}" 2>/dev/null || true
}
trap cleanup EXIT

# ---- Configuration ----
TRIALS="${TRIALS:-3}"
GPUS="${GPUS:-2}"
EPOCHS="${EPOCHS:-10}"
PROFILE_DURATION="${PROFILE_DURATION:-180}"
COOLDOWN="${COOLDOWN:-30}"
MASTER_PORT="${MASTER_PORT:-29501}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- Resolve Python and torchrun from the venv ----
VENV_DIR="${SCRIPT_DIR}/venv"
if [[ -f "${VENV_DIR}/bin/python3" ]]; then
    PYTHON="${VENV_DIR}/bin/python3"
    TORCHRUN="${VENV_DIR}/bin/torchrun"
    echo "Using venv Python: ${PYTHON}"
else
    PYTHON="$(which python3)"
    TORCHRUN="$(which torchrun 2>/dev/null || echo torchrun)"
    echo "WARNING: No venv found at ${VENV_DIR}, using system python3: ${PYTHON}"
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --trials) TRIALS="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --duration) PROFILE_DURATION="$2"; shift 2 ;;
        --cooldown) COOLDOWN="$2"; shift 2 ;;
        --native-only) NATIVE_ONLY=1; shift ;;
        --container-only) CONTAINER_ONLY=1; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "  MULTI-TRIAL EXPERIMENT RUNNER"
echo "  Trials: ${TRIALS} | GPUs: ${GPUS} | Epochs: ${EPOCHS}"
echo "  Profile Duration: ${PROFILE_DURATION}s | Cooldown: ${COOLDOWN}s"
echo "============================================================"

# ---- Check root ----
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: eBPF profilers require root privileges."
    echo "Run with: sudo ./21_run_experiment.sh"
    exit 1
fi

# ---- Collect system info ----
echo ""
echo "Collecting system information..."
SYSINFO_FILE="${SCRIPT_DIR}/results/system_info.json"
mkdir -p "${SCRIPT_DIR}/results"

${PYTHON} -c "
import json, subprocess, platform, os

info = {
    'hostname': platform.node(),
    'kernel': platform.release(),
    'os': platform.platform(),
    'cpu_model': open('/proc/cpuinfo').read().split('model name')[1].split('\n')[0].strip(': ') if 'model name' in open('/proc/cpuinfo').read() else 'unknown',
    'cpu_count': os.cpu_count(),
    'memory_gb': round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024.**3), 1),
}

try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], capture_output=True, text=True, timeout=5)
    info['gpus'] = [line.strip() for line in result.stdout.strip().split('\n')]
except:
    info['gpus'] = ['unknown']

try:
    result = subprocess.run(['docker', '--version'], capture_output=True, text=True, timeout=5)
    info['docker_version'] = result.stdout.strip()
except:
    info['docker_version'] = 'unknown'

try:
    import torch
    info['pytorch_version'] = torch.__version__
    info['cuda_version'] = torch.version.cuda
except:
    pass

with open('${SYSINFO_FILE}', 'w') as f:
    json.dump(info, f, indent=2)
print('System info saved to ${SYSINFO_FILE}')
print(json.dumps(info, indent=2))
"

# ---- GPU Reset / Cooldown Function ----
gpu_cooldown() {
    echo ""
    echo ">>> GPU cooldown: waiting ${COOLDOWN}s for thermals to stabilize..."
    # Reset GPU clocks to default
    nvidia-smi -rac 2>/dev/null || true
    sleep "${COOLDOWN}"
    # Show GPU temps after cooldown
    echo ">>> GPU temperatures after cooldown:"
    nvidia-smi --query-gpu=index,temperature.gpu,power.draw --format=csv,noheader
    echo ""
}

# ---- Run Native Trials ----
if [[ -z "${CONTAINER_ONLY}" ]]; then
    echo ""
    echo "============================================================"
    echo "  PHASE 1: NATIVE (BARE-METAL) TRIALS"
    echo "============================================================"

    for trial in $(seq 1 "${TRIALS}"); do
        TRIAL_DIR="results/native/trial_${trial}"
        mkdir -p "${TRIAL_DIR}"

        echo ""
        echo "------------------------------------------------------------"
        echo "  NATIVE TRIAL ${trial}/${TRIALS}"
        echo "------------------------------------------------------------"

        # GPU cooldown before each trial
        gpu_cooldown

        # Start profilers
        echo "[1/5] Starting CPU profiler..."
        ${PYTHON} "${SCRIPT_DIR}/21_cpu_profiler.py" \
            --duration "${PROFILE_DURATION}" \
            --output "${TRIAL_DIR}/21_cpu_results.csv" &
        PID_CPU=$!

        echo "[2/5] Starting syscall counter..."
        ${PYTHON} "${SCRIPT_DIR}/21_syscall_counter.py" \
            --duration "${PROFILE_DURATION}" \
            --output "${TRIAL_DIR}/21_syscall_results.csv" &
        PID_SYSCALL=$!

        echo "[3/5] Starting network profiler..."
        ${PYTHON} "${SCRIPT_DIR}/21_net_profiler.py" \
            --duration "${PROFILE_DURATION}" \
            --output "${TRIAL_DIR}/21_net_results.csv" &
        PID_NET=$!

        echo "[4/5] Starting GPU monitor..."
        ${PYTHON} "${SCRIPT_DIR}/21_gpu_monitor.py" \
            --duration "${PROFILE_DURATION}" \
            --interval 0.1 \
            --output "${TRIAL_DIR}/21_gpu_results.csv" &
        PID_GPU=$!
        PROFILER_PIDS="${PID_CPU} ${PID_SYSCALL} ${PID_NET} ${PID_GPU}"

        sleep 3

        echo "[5/5] Starting ML workload (native, trial ${trial})..."
        if [ "${GPUS}" -eq 1 ]; then
            ${PYTHON} "${SCRIPT_DIR}/21_ml_workload.py" \
                --gpus 1 \
                --epochs "${EPOCHS}" \
                --output "${TRIAL_DIR}/21_training_native.json"
        else
            ${TORCHRUN} --nproc_per_node="${GPUS}" --master_port="${MASTER_PORT}" \
                "${SCRIPT_DIR}/21_ml_workload.py" \
                --gpus "${GPUS}" \
                --epochs "${EPOCHS}" \
                --output "${TRIAL_DIR}/21_training_native.json"
        fi

        # Stop profilers
        kill ${PID_CPU} ${PID_SYSCALL} ${PID_NET} ${PID_GPU} 2>/dev/null || true
        wait ${PID_CPU} ${PID_SYSCALL} ${PID_NET} ${PID_GPU} 2>/dev/null || true
        PROFILER_PIDS=""

        echo ">>> Native trial ${trial} complete. Results in ${TRIAL_DIR}/"
    done

    # ---- Select median trial and copy to parent dir for analysis scripts ----
    echo ""
    echo ">>> Selecting median native trial..."
    ${PYTHON} -c "
import json, os, shutil
trials = []
for t in range(1, ${TRIALS}+1):
    jf = 'results/native/trial_{}/21_training_native.json'.format(t)
    if os.path.exists(jf):
        with open(jf) as f:
            d = json.load(f)
        trials.append((t, d.get('total_time', d.get('total_training_time', 999))))
if not trials:
    print('  WARNING: No native trials found')
else:
    trials.sort(key=lambda x: x[1])
    median_trial = trials[len(trials)//2][0]
    src = 'results/native/trial_{}'.format(median_trial)
    print('  Median trial: {} (time={:.1f}s)'.format(median_trial, trials[len(trials)//2][1]))
    for f in os.listdir(src):
        shutil.copy2(os.path.join(src, f), os.path.join('results/native', f))
    print('  Copied trial {} data to results/native/'.format(median_trial))
"
fi

# ---- Run Container Trials ----
if [[ -z "${NATIVE_ONLY}" ]]; then
    echo ""
    echo "============================================================"
    echo "  PHASE 2: CONTAINERIZED (DOCKER BRIDGE) TRIALS"
    echo "============================================================"

    # Check Docker image
    IMAGE_NAME="group21-ml-profiling"
    CONTAINER_NAME="group21-profiled-run"

    if ! docker image inspect "${IMAGE_NAME}" &>/dev/null; then
        echo "ERROR: Docker image '${IMAGE_NAME}' not found."
        echo "Build it first: ./G_21_container_setup.sh build"
        exit 1
    fi

    for trial in $(seq 1 "${TRIALS}"); do
        TRIAL_DIR="results/container/trial_${trial}"
        mkdir -p "${TRIAL_DIR}"

        echo ""
        echo "------------------------------------------------------------"
        echo "  CONTAINER TRIAL ${trial}/${TRIALS}"
        echo "------------------------------------------------------------"

        # GPU cooldown before each trial
        gpu_cooldown

        # Clean up any existing container
        docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

        # Start profilers on HOST
        echo "[1/5] Starting CPU profiler (host)..."
        ${PYTHON} "${SCRIPT_DIR}/21_cpu_profiler.py" \
            --duration "${PROFILE_DURATION}" \
            --output "${TRIAL_DIR}/21_cpu_results.csv" &
        PID_CPU=$!

        echo "[2/5] Starting syscall counter (host)..."
        ${PYTHON} "${SCRIPT_DIR}/21_syscall_counter.py" \
            --duration "${PROFILE_DURATION}" \
            --output "${TRIAL_DIR}/21_syscall_results.csv" &
        PID_SYSCALL=$!

        echo "[3/5] Starting network profiler (host)..."
        ${PYTHON} "${SCRIPT_DIR}/21_net_profiler.py" \
            --duration "${PROFILE_DURATION}" \
            --output "${TRIAL_DIR}/21_net_results.csv" &
        PID_NET=$!

        echo "[4/5] Starting GPU monitor (host)..."
        ${PYTHON} "${SCRIPT_DIR}/21_gpu_monitor.py" \
            --duration "${PROFILE_DURATION}" \
            --interval 0.1 \
            --output "${TRIAL_DIR}/21_gpu_results.csv" &
        PID_GPU=$!
        PROFILER_PIDS="${PID_CPU} ${PID_SYSCALL} ${PID_NET} ${PID_GPU}"

        sleep 3

        echo "[5/5] Starting ML workload (container, trial ${trial})..."
        set +e  # Don't abort on docker failure
        if [ "${GPUS}" -eq 1 ]; then
            docker run \
                --name "${CONTAINER_NAME}" \
                --gpus all \
                --shm-size=2g \
                --ulimit memlock=-1 \
                --network=bridge \
                -v "${SCRIPT_DIR}/results:/workspace/results" \
                -v "${SCRIPT_DIR}/data:/workspace/data" \
                "${IMAGE_NAME}" \
                python3 21_ml_workload.py \
                    --gpus 1 \
                    --epochs "${EPOCHS}" \
                    --output "results/container/trial_${trial}/21_training_container.json"
        else
            docker run \
                --name "${CONTAINER_NAME}" \
                --gpus all \
                --shm-size=2g \
                --ulimit memlock=-1 \
                --network=bridge \
                -v "${SCRIPT_DIR}/results:/workspace/results" \
                -v "${SCRIPT_DIR}/data:/workspace/data" \
                "${IMAGE_NAME}" \
                bash -c "torchrun --nproc_per_node=${GPUS} --master_port=${MASTER_PORT} 21_ml_workload.py \
                    --gpus ${GPUS} \
                    --epochs ${EPOCHS} \
                    --output results/container/trial_${trial}/21_training_container.json"
        fi
        DOCKER_EXIT=$?
        set -e

        if [ ${DOCKER_EXIT} -ne 0 ]; then
            echo "ERROR: Container trial ${trial} failed (exit code ${DOCKER_EXIT})"
        fi

        # Stop profilers
        kill ${PID_CPU} ${PID_SYSCALL} ${PID_NET} ${PID_GPU} 2>/dev/null || true
        wait ${PID_CPU} ${PID_SYSCALL} ${PID_NET} ${PID_GPU} 2>/dev/null || true
        PROFILER_PIDS=""

        # Capture container metadata
        docker inspect "${CONTAINER_NAME}" > "${TRIAL_DIR}/21_container_inspect.json" 2>/dev/null || true
        docker rm "${CONTAINER_NAME}" 2>/dev/null || true

        echo ">>> Container trial ${trial} complete. Results in ${TRIAL_DIR}/"
    done

    # ---- Select median container trial and copy to parent dir ----
    echo ""
    echo ">>> Selecting median container trial..."
    ${PYTHON} -c "
import json, os, shutil
trials = []
for t in range(1, ${TRIALS}+1):
    jf = 'results/container/trial_{}/21_training_container.json'.format(t)
    if os.path.exists(jf):
        with open(jf) as f:
            d = json.load(f)
        trials.append((t, d.get('total_time', d.get('total_training_time', 999))))
if not trials:
    print('  WARNING: No container trials completed successfully')
else:
    trials.sort(key=lambda x: x[1])
    median_trial = trials[len(trials)//2][0]
    src = 'results/container/trial_{}'.format(median_trial)
    print('  Median trial: {} (time={:.1f}s)'.format(median_trial, trials[len(trials)//2][1]))
    for f in os.listdir(src):
        shutil.copy2(os.path.join(src, f), os.path.join('results/container', f))
    print('  Copied trial {} data to results/container/'.format(median_trial))
"
fi

echo ""
echo "============================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "============================================================"
echo ""
echo "Results structure:"
find results/ -name "*.json" -o -name "*.csv" | head -40
echo ""
echo "Next step: ${PYTHON} G_21_compare_results.py"
echo "============================================================"
