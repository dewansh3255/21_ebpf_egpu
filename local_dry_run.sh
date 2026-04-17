#!/bin/bash

echo "==========================================="
echo "  Automated Local Dry-Run (Phases 1, 2, 3) "
echo "==========================================="

# 1. Ensure monitor scripts are executable
chmod +x monitor/monitor_network.py
chmod +x monitor/monitor_gpu.py

# 2. Start eBPF Monitors
echo "[+] Starting eBPF Monitors in the background..."
sudo python3 monitor/monitor_network.py > /dev/null 2>&1 &
NET_PID=$!
sudo python3 monitor/monitor_gpu.py > /dev/null 2>&1 &
GPU_PID=$!

# Wait for eBPF compilers to finish and attach to the kernel
echo "[+] Waiting 8 seconds for BPF to compile and attach to the kernel..."
sleep 8

# 3. Start PyTorch Server
echo "[+] Starting Parameter Server (Port 8000)..."
python3 src/server.py > /dev/null 2>&1 &
SERVER_PID=$!
sleep 3 # Give server time to boot up

# 4. Start Local Clients
echo "[+] Starting Client 1..."
python3 src/client.py 1 &
CLIENT1_PID=$!

echo "[+] Starting Client 2 (Local Simulation)..."
python3 src/client.py 2 &
CLIENT2_PID=$!

# 5. Wait for both clients to finish training
echo "[+] Waiting for Federated Learning to complete..."
wait $CLIENT1_PID
wait $CLIENT2_PID

echo "[✓] Training complete. Stopping monitors and server..."

# 6. Cleanup Background Processes (SIGINT ensures CSVs save properly)
sudo kill -SIGINT $NET_PID
sudo kill -SIGINT $GPU_PID
kill $SERVER_PID

# Wait a moment for Python to flush the CSV buffers to the hard drive
sleep 3

# 7. Find the latest CSV files automatically
echo "[+] Locating newly generated CSV files..."
LATEST_NET_CSV=$(ls -t monitor/network_trace_PHASE1_*.csv | head -1)
LATEST_GPU_CSV=$(ls -t monitor/gpu_trace_PHASE2_*.csv | head -1)

if [[ -n "$LATEST_NET_CSV" && -n "$LATEST_GPU_CSV" ]]; then
    echo "[+] Found Network Trace: $LATEST_NET_CSV"
    echo "[+] Found GPU Trace: $LATEST_GPU_CSV"
    
    # 8. Generate the Unified Timeline Dashboard
    echo "[+] Generating Unified Timeline Dashboard..."
    python3 monitor/plot_unified_timeline.py "$LATEST_NET_CSV" "$LATEST_GPU_CSV"
    
    echo "==========================================="
    echo "  SUCCESS! Open phase3_unified_timeline.png"
    echo "==========================================="
else
    echo "[!] Error: Could not find the generated CSV files."
fi