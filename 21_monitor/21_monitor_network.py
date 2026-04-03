# monitor_network.py
from bcc import BPF
import time
import socket
import struct
import csv
import os

bpf_text = """
// --- UBUNTU KERNEL HEADER BUG WORKAROUND ---
#define BPF_LOAD_ACQ   (1 << 4)
#define BPF_STORE_REL  (1 << 5)
struct bpf_wq { void *work; };
// -------------------------------------------
#include <uapi/linux/ptrace.h>
#include <net/sock.h>
#include <bcc/proto.h>

struct data_t {
    u64 ts_us;
    u32 pid;
    u32 tid;
    char comm[TASK_COMM_LEN];
    u32 saddr;
    u32 daddr;
    u16 lport;
    u16 dport;
    u64 size;
    u32 is_receive;
};

BPF_PERF_OUTPUT(events);

int trace_tcp_sendmsg(struct pt_regs *ctx, struct sock *sk, struct msghdr *msg, size_t size) {
    u16 dport = sk->sk_dport;
    if (ntohs(dport) == 8000 || sk->sk_num == 8000) {
        struct data_t data = {};
        u64 pid_tgid = bpf_get_current_pid_tgid();
        data.pid = pid_tgid >> 32;
        data.tid = pid_tgid;
        bpf_get_current_comm(&data.comm, sizeof(data.comm));
        
        data.ts_us = bpf_ktime_get_ns() / 1000;
        data.saddr = sk->sk_rcv_saddr;
        data.daddr = sk->sk_daddr;
        data.lport = sk->sk_num;
        data.dport = ntohs(dport);
        data.size = size;
        data.is_receive = 0;
        events.perf_submit(ctx, &data, sizeof(data));
    }
    return 0;
}

int trace_tcp_recvmsg(struct pt_regs *ctx, struct sock *sk, struct msghdr *msg, size_t len, int nonblock, int flags, int *addr_len) {
    u16 dport = sk->sk_dport;
    if (ntohs(dport) == 8000 || sk->sk_num == 8000) {
        struct data_t data = {};
        u64 pid_tgid = bpf_get_current_pid_tgid();
        data.pid = pid_tgid >> 32;
        data.tid = pid_tgid;
        bpf_get_current_comm(&data.comm, sizeof(data.comm));
        
        data.ts_us = bpf_ktime_get_ns() / 1000;
        data.saddr = sk->sk_daddr; 
        data.daddr = sk->sk_rcv_saddr;
        data.lport = sk->sk_num;
        data.dport = ntohs(dport);
        data.size = len;
        data.is_receive = 1;
        events.perf_submit(ctx, &data, sizeof(data));
    }
    return 0;
}
"""

# CSV Setup
csv_filename = f"network_trace_{int(time.time())}.csv"
csv_file = open(csv_filename, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp(us)", "Time", "Direction", "PID", "TID", "Command", "Src_IP", "Dest_IP", "Port", "Payload_Size_Bytes"])

print("Compiling Advanced Network Analyzer with CSV Logging...")
b = BPF(text=bpf_text)
b.attach_kprobe(event="tcp_sendmsg", fn_name="trace_tcp_sendmsg")
b.attach_kprobe(event="tcp_recvmsg", fn_name="trace_tcp_recvmsg")

print(f"Logging all raw traffic to: {csv_filename}")
print(f"{'TIME':<10} {'DIRECTION':<10} {'SRC IP':<16} {'DEST IP':<16} {'SIZE (Bytes)'}")
print("-" * 65)

def print_event(cpu, data, size):
    event = b["events"].event(data)
    try:
        src_ip = socket.inet_ntoa(struct.pack("<L", event.saddr))
        dest_ip = socket.inet_ntoa(struct.pack("<L", event.daddr))
        time_str = time.strftime("%H:%M:%S")
        direction = "IN_RECV" if event.is_receive else "OUT_SEND"
        comm = event.comm.decode('utf-8', 'replace')
        
        # Write EVERY event to CSV for deep analysis
        csv_writer.writerow([event.ts_us, time_str, direction, event.pid, event.tid, comm, src_ip, dest_ip, event.dport, event.size])
        csv_file.flush()
        
        # Only print large payloads to terminal to keep it readable
        if event.size > 1000:
            print(f"{time_str:<10} {direction:<10} {src_ip:<16} {dest_ip:<16} {event.size}")
    except:
        pass

b["events"].open_perf_buffer(print_event)
try:
    while True:
        b.perf_buffer_poll()
except KeyboardInterrupt:
    print("\nDetaching Network Monitor. File saved.")
    csv_file.close()