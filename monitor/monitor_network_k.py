from bcc import BPF
import time
import socket
import struct
import csv

bpf_text = """
// Force load kernel configs to fix the Ubuntu ns_common.h bug
#include <linux/kconfig.h>

#include <uapi/linux/ptrace.h>
#include <net/sock.h>
#include <bcc/proto.h>
#include <linux/types.h>

#define BSWAP16(x) __builtin_bswap16(x)

BPF_PERF_OUTPUT(network_events);

struct data_t {
    u32 pid;
    u32 saddr;
    u32 daddr;
    u16 lport;
    u16 dport;
    u64 timestamp;
    u32 size; // ADDED: Payload size!
    char comm[TASK_COMM_LEN];
    int is_recv;
};

// Added arguments to capture the size parameter
int trace_tcp_sendmsg(struct pt_regs *ctx, struct sock *sk, struct msghdr *msg, size_t size) {
    u16 dport = sk->sk_dport;
    
    if (BSWAP16(dport) == 8000 || sk->sk_num == 8000) {
        struct data_t data = {};
        data.pid = bpf_get_current_pid_tgid() >> 32;
        bpf_get_current_comm(data.comm, sizeof(data.comm));
        data.timestamp = bpf_ktime_get_ns();
        data.saddr = sk->sk_rcv_saddr;
        data.daddr = sk->sk_daddr;
        data.lport = sk->sk_num;
        data.dport = BSWAP16(dport);
        data.size = size; // Save the size
        data.is_recv = 0;
        network_events.perf_submit(ctx, &data, sizeof(data));
    }
    return 0;
}

// Added arguments to capture the len parameter
int trace_tcp_recvmsg(struct pt_regs *ctx, struct sock *sk, struct msghdr *msg, size_t len) {
    u16 dport = sk->sk_dport;
    
    if (BSWAP16(dport) == 8000 || sk->sk_num == 8000) {
        struct data_t data = {};
        data.pid = bpf_get_current_pid_tgid() >> 32;
        bpf_get_current_comm(data.comm, sizeof(data.comm));
        data.timestamp = bpf_ktime_get_ns();
        data.saddr = sk->sk_rcv_saddr;
        data.daddr = sk->sk_daddr;
        data.lport = sk->sk_num;
        data.dport = BSWAP16(dport);
        data.size = len; // Save the size
        data.is_recv = 1;
        network_events.perf_submit(ctx, &data, sizeof(data));
    }
    return 0;
}
"""

# CSV Setup
csv_filename = f"network_trace_{int(time.time())}.csv"
csv_file = open(csv_filename, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp(ns)", "Time", "Direction", "PID", "Command", "Src_IP", "Dest_IP", "Port", "Payload_Size_Bytes"])

print("Compiling Advanced Network Analyzer with CSV Logging...")

# Initialize BPF and load C code
b = BPF(text=bpf_text, cflags=["-Wno-error", "-Wno-array-bounds", "-Wno-macro-redefined", "-Wno-microsoft-anon-tag", "-fms-extensions"])

# Attach to the kernel's TCP send function
b.attach_kprobe(event="tcp_sendmsg", fn_name="trace_tcp_sendmsg")

# Attach to the kernel's TCP receive function
b.attach_kprobe(event="tcp_recvmsg", fn_name="trace_tcp_recvmsg")

print(f"Logging all raw traffic to: {csv_filename}")
print(f"{'TIME':<10} {'DIRECTION':<10} {'SRC IP':<16} {'DEST IP':<16} {'SIZE (Bytes)'}")
print("-" * 65)

def print_event(cpu, data, size):
    # This try/except block will now actually print the error if it fails!
    try:
        event = b["network_events"].event(data)
        src_ip = socket.inet_ntoa(struct.pack("<L", event.saddr))
        dest_ip = socket.inet_ntoa(struct.pack("<L", event.daddr))
        time_str = time.strftime("%H:%M:%S")
        
        # Fixed the variable name to match C struct
        direction = "IN_RECV" if event.is_recv else "OUT_SEND"
        comm = event.comm.decode('utf-8', 'replace')
        
        # Write EVERY event to CSV (Fixed the variable names!)
        csv_writer.writerow([event.timestamp, time_str, direction, event.pid, comm, src_ip, dest_ip, event.dport, event.size])
        csv_file.flush()
        
        # Print ALL events to the terminal so we can finally see them!
        print(f"{time_str:<10} {direction:<10} {src_ip:<16} {dest_ip:<16} {event.size}")
        
    except Exception as e:
        print(f"DEBUG - Event processing failed: {e}")

b["network_events"].open_perf_buffer(print_event)

try:
    while True:
        b.perf_buffer_poll()
except KeyboardInterrupt:
    print("\nDetaching Network Monitor. File saved.")
    csv_file.close()