# monitor_gpu.py
from bcc import BPF
import time
import csv

bpf_text = """
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

struct data_t {
    u64 ts_us;
    u32 pid;
    u32 tid;
    u32 cpu;
    char comm[TASK_COMM_LEN];
    u64 ioctl_cmd;
    u64 fd;
};

BPF_PERF_OUTPUT(gpu_events);

int trace_sys_ioctl(struct pt_regs *ctx, unsigned int fd, unsigned int cmd, unsigned long arg) {
    char comm[TASK_COMM_LEN];
    bpf_get_current_comm(&comm, sizeof(comm));
    
    // Catch PyTorch execution
    if (comm[0] == 'p' && comm[1] == 'y' && comm[2] == 't') {
        struct data_t data = {};
        u64 pid_tgid = bpf_get_current_pid_tgid();
        data.pid = pid_tgid >> 32;
        data.tid = pid_tgid;
        data.cpu = bpf_get_smp_processor_id();
        data.ts_us = bpf_ktime_get_ns() / 1000;
        data.ioctl_cmd = cmd;
        data.fd = fd;
        bpf_get_current_comm(&data.comm, sizeof(data.comm));
        
        gpu_events.perf_submit(ctx, &data, sizeof(data));
    }
    return 0;
}
"""

# CSV Setup
csv_filename = f"gpu_ioctl_trace_{int(time.time())}.csv"
csv_file = open(csv_filename, 'w', newline='')
csv_writer = csv.writer(csv_file)
# Header includes deep IOCTL decoding parameters
csv_writer.writerow(["Timestamp(us)", "Time", "CPU_Core", "PID", "TID", "Command", "FD", "Raw_Cmd_Hex", "IOCTL_Dir", "IOCTL_Type(Char)", "IOCTL_Nr", "Data_Size_Bytes"])

print("Compiling Deep GPU IOCTL Analyzer with CSV Logging...")
b = BPF(text=bpf_text)
syscall_name = b.get_syscall_fnname("ioctl")
b.attach_kprobe(event=syscall_name, fn_name="trace_sys_ioctl")

print(f"Logging raw high-frequency data to: {csv_filename}")
print(f"{'TIME':<10} {'PID':<8} {'COMMAND':<12} {'SUMMARY'}")
print("-" * 55)

cmd_count = 0
start_time = time.time()

def decode_ioctl(cmd):
    """Breaks down the Linux IOCTL binary structure."""
    direction = (cmd >> 30) & 0x03
    size = (cmd >> 16) & 0x3FFF
    type_code = (cmd >> 8) & 0xFF
    nr = cmd & 0xFF
    
    dir_str = "NONE"
    if direction == 1: dir_str = "WRITE"
    elif direction == 2: dir_str = "READ"
    elif direction == 3: dir_str = "READ/WRITE"
    
    type_char = chr(type_code) if 32 <= type_code <= 126 else f"0x{type_code:02x}"
    return dir_str, type_char, nr, size

def print_gpu_event(cpu, data, size):
    global cmd_count, start_time
    event = b["gpu_events"].event(data)
    
    time_str = time.strftime("%H:%M:%S")
    comm = event.comm.decode('utf-8', 'replace')
    
    # Deep Decode the IOCTL command
    dir_str, type_char, nr, data_size = decode_ioctl(event.ioctl_cmd)
    
    # Write EVERY single hardware command to the CSV
    csv_writer.writerow([
        event.ts_us, time_str, event.cpu, event.pid, event.tid, 
        comm, event.fd, hex(event.ioctl_cmd), dir_str, type_char, nr, data_size
    ])
    
    # Console summary logic (Prevent terminal lag)
    cmd_count += 1
    if time.time() - start_time > 1.0:
        print(f"{time_str:<10} {event.pid:<8} {comm:<12} {cmd_count} IOCTLs/sec processed")
        csv_file.flush()
        cmd_count = 0
        start_time = time.time()

b["gpu_events"].open_perf_buffer(print_gpu_event)
try:
    while True:
        b.perf_buffer_poll()
except KeyboardInterrupt:
    print("\nDetaching GPU Monitor. File saved.")
    csv_file.close()