#!/usr/bin/env python3
"""
G_21_plot_hardcoded.py - Hardcoded matplotlib plots for eBPF/eGPU experiment results.
Group 21 - Part B Final Submission

All data values are hardcoded from actual experiment results.
No CSV files are read as input.

Experiment: ResNet-18 on CIFAR-10, 2x NVIDIA H100 NVL GPUs, 10 epochs, batch_size=128
Comparison: Native vs Docker Container execution with eBPF profiling
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "plots_hardcoded")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_training_loss():
    """Plot 1: Training Loss per Epoch - Native vs Container"""
    epochs = list(range(1, 11))
    native_loss = [1.6715, 1.2365, 1.0123, 0.8746, 0.7588, 0.6792, 0.6193, 0.5620, 0.5347, 0.5196]
    container_loss = [1.6527, 1.2230, 0.9863, 0.8436, 0.7296, 0.6486, 0.5859, 0.5391, 0.5104, 0.4988]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, native_loss, 'o-', color='#2196F3', linewidth=2, markersize=8, label='Native')
    ax.plot(epochs, container_loss, 's--', color='#FF5722', linewidth=2, markersize=8, label='Container')
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Training Loss', fontsize=13)
    ax.set_title('Training Loss: Native vs Container', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "01_training_loss.png"), dpi=150)
    plt.close(fig)
    print("  [1/11] Training Loss plot saved")


def plot_training_accuracy():
    """Plot 2: Training & Test Accuracy per Epoch"""
    epochs = list(range(1, 11))
    native_train_acc = [37.24, 54.64, 63.83, 68.93, 73.31, 76.15, 78.35, 80.31, 81.28, 82.14]
    container_train_acc = [38.08, 55.61, 64.92, 69.97, 74.14, 77.14, 79.70, 81.08, 82.23, 82.65]
    native_test_acc = [47.21, 58.30, 65.14, 66.73, 72.45, 74.64, 76.34, 78.68, 80.11, 80.31]
    container_test_acc = [47.87, 60.12, 65.43, 69.64, 72.76, 76.53, 78.02, 79.42, 80.92, 81.19]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(epochs, native_train_acc, 'o-', color='#2196F3', linewidth=2, markersize=7, label='Native')
    ax1.plot(epochs, container_train_acc, 's--', color='#FF5722', linewidth=2, markersize=7, label='Container')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Accuracy (%)', fontsize=12)
    ax1.set_title('Training Accuracy', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)

    ax2.plot(epochs, native_test_acc, 'o-', color='#2196F3', linewidth=2, markersize=7, label='Native')
    ax2.plot(epochs, container_test_acc, 's--', color='#FF5722', linewidth=2, markersize=7, label='Container')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax2.set_title('Test Accuracy', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)

    fig.suptitle('Model Accuracy: Native vs Container', fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "02_training_accuracy.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [2/11] Training Accuracy plot saved")


def plot_throughput():
    """Plot 3: Throughput (samples/sec) per Epoch"""
    epochs = list(range(1, 11))
    native_throughput = [2053.7, 2205.3, 2265.6, 3120.8, 2348.9, 2180.9, 2490.9, 2322.4, 2292.4, 2399.6]
    container_throughput = [2126.3, 1948.1, 2220.5, 2600.8, 2480.3, 2195.4, 1920.6, 2495.8, 2716.8, 2301.5]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(epochs))
    width = 0.35
    bars1 = ax.bar(x - width/2, native_throughput, width, label='Native', color='#2196F3', alpha=0.85)
    bars2 = ax.bar(x + width/2, container_throughput, width, label='Container', color='#FF5722', alpha=0.85)
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Throughput (samples/sec)', fontsize=13)
    ax.set_title('Training Throughput: Native vs Container', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(epochs)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Add averages
    native_avg = np.mean(native_throughput)
    container_avg = np.mean(container_throughput)
    ax.axhline(y=native_avg, color='#2196F3', linestyle=':', alpha=0.7, label=f'Native avg: {native_avg:.0f}')
    ax.axhline(y=container_avg, color='#FF5722', linestyle=':', alpha=0.7, label=f'Container avg: {container_avg:.0f}')

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "03_throughput.png"), dpi=150)
    plt.close(fig)
    print("  [3/11] Throughput plot saved")


def plot_epoch_time():
    """Plot 4: Per-Epoch Training Time"""
    epochs = list(range(1, 11))
    native_time = [12.17, 11.34, 11.03, 8.01, 10.64, 11.46, 10.04, 10.76, 10.91, 10.42]
    container_time = [11.76, 12.83, 11.26, 9.61, 10.08, 11.39, 13.02, 10.02, 9.20, 10.86]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, native_time, 'o-', color='#2196F3', linewidth=2, markersize=8, label='Native')
    ax.plot(epochs, container_time, 's--', color='#FF5722', linewidth=2, markersize=8, label='Container')

    ax.axhline(y=np.mean(native_time), color='#2196F3', linestyle=':', alpha=0.6)
    ax.axhline(y=np.mean(container_time), color='#FF5722', linestyle=':', alpha=0.6)

    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Time (seconds)', fontsize=13)
    ax.set_title('Per-Epoch Training Time: Native vs Container', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "04_epoch_time.png"), dpi=150)
    plt.close(fig)
    print("  [4/11] Epoch Time plot saved")


def plot_total_time_comparison():
    """Plot 5: Total Training Time + Overhead Summary"""
    categories = ['Training\nTime (s)', 'Throughput\n(samples/s)', 'GPU Util\n(%)', 'GPU Power\n(W)']
    native_vals = [118.3, 2368, 99.9, 213.8]
    container_vals = [121.3, 2301, 99.9, 220.1]
    overhead_pct = [2.5, -2.8, 0.0, 2.9]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    for i, (cat, nv, cv, oh) in enumerate(zip(categories, native_vals, container_vals, overhead_pct)):
        ax = axes[i]
        bars = ax.bar(['Native', 'Container'], [nv, cv],
                      color=['#2196F3', '#FF5722'], alpha=0.85, width=0.5)
        ax.set_title(cat, fontsize=11, fontweight='bold')
        for bar, val in zip(bars, [nv, cv]):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        oh_color = '#d32f2f' if oh > 0 else '#388e3c'
        oh_sign = '+' if oh >= 0 else ''
        ax.text(0.5, 0.02, f'Overhead: {oh_sign}{oh}%', transform=ax.transAxes,
                ha='center', fontsize=10, color=oh_color, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Overall Performance: Native vs Container', fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "05_total_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [5/11] Total Comparison plot saved")


def plot_gpu_utilization():
    """Plot 6: GPU Utilization & Power over time (sampled summary)"""
    # Sampled every ~10s from 1619 native records and 1661 container records
    # GPU 0 data sampled at regular intervals
    native_time_offsets = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 118]
    native_gpu_util = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    native_power = [151.4, 188.2, 210.5, 225.3, 218.7, 215.2, 220.8, 218.4, 216.9, 219.3, 217.5, 215.8, 210.2]
    native_temp = [81, 83, 84, 85, 85, 85, 86, 86, 86, 86, 86, 85, 85]

    container_time_offsets = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 121]
    container_gpu_util = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    container_power = [155.8, 192.4, 215.6, 228.1, 222.3, 219.8, 224.5, 221.6, 220.1, 222.8, 220.3, 218.5, 213.8]
    container_temp = [82, 84, 85, 86, 86, 86, 87, 87, 87, 87, 87, 86, 86]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

    # GPU Utilization
    axes[0].plot(native_time_offsets, native_gpu_util, '-', color='#2196F3', linewidth=2, label='Native GPU 0')
    axes[0].plot(container_time_offsets, container_gpu_util, '--', color='#FF5722', linewidth=2, label='Container GPU 0')
    axes[0].set_ylabel('GPU Utilization (%)', fontsize=12)
    axes[0].set_title('GPU Utilization Over Time', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].set_ylim(95, 102)
    axes[0].grid(True, alpha=0.3)

    # Power
    axes[1].plot(native_time_offsets, native_power, '-', color='#2196F3', linewidth=2, label='Native GPU 0')
    axes[1].plot(container_time_offsets, container_power, '--', color='#FF5722', linewidth=2, label='Container GPU 0')
    axes[1].set_ylabel('Power (W)', fontsize=12)
    axes[1].set_title('GPU Power Draw Over Time', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # Temperature
    axes[2].plot(native_time_offsets, native_temp, '-', color='#2196F3', linewidth=2, label='Native GPU 0')
    axes[2].plot(container_time_offsets, container_temp, '--', color='#FF5722', linewidth=2, label='Container GPU 0')
    axes[2].set_ylabel('Temperature (°C)', fontsize=12)
    axes[2].set_xlabel('Time (seconds)', fontsize=12)
    axes[2].set_title('GPU Temperature Over Time', fontsize=13, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('GPU Metrics: Native vs Container (NVIDIA H100 NVL)', fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "06_gpu_metrics.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [6/11] GPU Metrics plot saved")


def plot_syscall_comparison():
    """Plot 7: Top Syscalls - Native vs Container"""
    # Top 10 syscalls by count
    syscall_names = ['gettid', 'poll', 'futex', 'read', 'close',
                     'newfstatat', 'write', 'openat', 'ioctl', 'accept']
    native_counts = [3609347, 2960629, 463068, 395931, 232337,
                     210549, 183384, 142452, 126003, 104299]
    container_counts = [3774924, 3068178, 394834, 385357, 248747,
                        217360, 178427, 157760, 128876, 107566]

    fig, ax = plt.subplots(figsize=(12, 7))
    y = np.arange(len(syscall_names))
    height = 0.35
    bars1 = ax.barh(y - height/2, [c/1e6 for c in native_counts], height,
                    label='Native', color='#2196F3', alpha=0.85)
    bars2 = ax.barh(y + height/2, [c/1e6 for c in container_counts], height,
                    label='Container', color='#FF5722', alpha=0.85)
    ax.set_xlabel('Count (millions)', fontsize=13)
    ax.set_ylabel('Syscall', fontsize=13)
    ax.set_title('Top 10 Syscalls: Native vs Container', fontsize=15, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(syscall_names, fontsize=11)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')

    # Add total counts annotation
    ax.text(0.98, 0.02, f'Total: Native={9521613:,}  Container={9766704:,}  (+2.6%)',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "07_syscall_comparison.png"), dpi=150)
    plt.close(fig)
    print("  [7/11] Syscall Comparison plot saved")


def plot_syscall_latency():
    """Plot 8: Syscall Average Latency Comparison"""
    syscall_names = ['gettid', 'poll', 'futex', 'read', 'close',
                     'newfstatat', 'write', 'openat', 'ioctl', 'accept']
    # Average latency in microseconds
    native_avg_us = [0.368, 849632.7, 3619621.8, 17.943, 0.941,
                     3.030, 7.124, 129.052, 108.526, 9.590]
    container_avg_us = [0.375, 1022983.8, 4755975.1, 356070.2, 0.931,
                        4.249, 8.928, 122.247, 108.692, 9.742]

    # Only plot non-blocking syscalls (exclude poll, futex, epoll which have huge waits)
    plot_names = ['gettid', 'read', 'close', 'newfstatat', 'write', 'openat', 'ioctl', 'accept']
    plot_native = [0.368, 17.943, 0.941, 3.030, 7.124, 129.052, 108.526, 9.590]
    plot_container = [0.375, 356.070, 0.931, 4.249, 8.928, 122.247, 108.692, 9.742]

    fig, ax = plt.subplots(figsize=(12, 6))
    y = np.arange(len(plot_names))
    height = 0.35
    ax.barh(y - height/2, plot_native, height, label='Native', color='#2196F3', alpha=0.85)
    ax.barh(y + height/2, plot_container, height, label='Container', color='#FF5722', alpha=0.85)
    ax.set_xlabel('Average Latency (μs)', fontsize=13)
    ax.set_ylabel('Syscall', fontsize=13)
    ax.set_title('Syscall Average Latency (Non-Blocking): Native vs Container', fontsize=14, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(plot_names, fontsize=11)
    ax.set_xscale('log')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "08_syscall_latency.png"), dpi=150)
    plt.close(fig)
    print("  [8/11] Syscall Latency plot saved")


def plot_network_comparison():
    """Plot 9: Network I/O - TCP Send/Recv Stats"""
    categories = ['TCP Send\nAvg Lat (μs)', 'TCP Recv\nAvg Lat (μs)', 'TCP Send\nCount', 'TCP Recv\nCount']
    native_vals = [38.48, 5.95, 11445, 17959]
    container_vals = [34.35, 6.72, 18007, 22127]
    overhead_pct = [-10.7, 12.8, None, None]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    for i, (cat, nv, cv) in enumerate(zip(categories, native_vals, container_vals)):
        ax = axes[i]
        bars = ax.bar(['Native', 'Container'], [nv, cv],
                      color=['#2196F3', '#FF5722'], alpha=0.85, width=0.5)
        ax.set_title(cat, fontsize=11, fontweight='bold')
        for bar, val in zip(bars, [nv, cv]):
            fmt = f'{val:.2f}' if val < 100 else f'{val:,.0f}'
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    fmt, ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Network I/O: Native vs Container (eBPF TCP Profiling)', fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "09_network_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [9/11] Network Comparison plot saved")


def plot_sched_latency():
    """Plot 10: CPU Scheduler Latency - eBPF profiled"""
    metrics = ['Mean\nLatency (μs)', 'P95\nLatency (μs)', 'P99\nLatency (μs)']
    native_vals = [19.71, 12.27, 19.05]
    container_vals = [18.07, 12.37, 19.55]
    overhead_pct = [-8.3, 0.8, 2.6]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart
    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax1.bar(x - width/2, native_vals, width, label='Native', color='#2196F3', alpha=0.85)
    bars2 = ax1.bar(x + width/2, container_vals, width, label='Container', color='#FF5722', alpha=0.85)
    ax1.set_ylabel('Latency (μs)', fontsize=13)
    ax1.set_title('Scheduler Latency Distribution', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=11)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars1, native_vals):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, container_vals):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val}', ha='center', va='bottom', fontsize=9)

    # Context switches
    ctx_data = {
        'Native': {'total': 5825652, 'per_sec': 44974, 'ml_pct': 88.13},
        'Container': {'total': 5993381, 'per_sec': 46024, 'ml_pct': 87.45},
    }
    labels = list(ctx_data.keys())
    totals = [ctx_data[l]['total'] / 1e6 for l in labels]
    per_sec = [ctx_data[l]['per_sec'] for l in labels]

    bars = ax2.bar(labels, totals, color=['#2196F3', '#FF5722'], alpha=0.85, width=0.5)
    ax2.set_ylabel('Total Context Switches (millions)', fontsize=12)
    ax2.set_title('Context Switches (eBPF Tracepoints)', fontsize=13, fontweight='bold')
    for bar, total, ps in zip(bars, totals, per_sec):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{total:.2f}M\n({ps:,}/s)', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    overhead = (ctx_data['Container']['total'] - ctx_data['Native']['total']) / ctx_data['Native']['total'] * 100
    ax2.text(0.5, 0.02, f'Container overhead: +{overhead:.1f}%', transform=ax2.transAxes,
             ha='center', fontsize=11, color='#d32f2f', fontweight='bold')

    fig.suptitle('CPU Scheduling: Native vs Container (eBPF Profiling)', fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "10_sched_latency.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [10/11] Scheduler Latency plot saved")


def plot_overhead_summary():
    """Plot 11: Overall Overhead Summary Table"""
    metrics = [
        'Training Time (sec)', 'Throughput (samples/sec)', 'Sched Latency Mean (μs)',
        'Sched Latency P95 (μs)', 'Sched Latency P99 (μs)', 'Total Syscalls',
        'Unique Syscall Types', 'TCP Send Avg Lat (μs)', 'TCP Send Count',
        'TCP Recv Avg Lat (μs)', 'TCP Recv Count', 'GPU Avg Utilization (%)',
        'GPU Avg Power (W)'
    ]
    native_vals = ['118.3', '2,368', '19.71', '12.27', '19.05', '9,521,613',
                   '122', '38.48', '11,445', '5.95', '17,959', '99.9', '213.8']
    container_vals = ['121.3', '2,301', '18.07', '12.37', '19.55', '9,766,704',
                      '172', '34.35', '18,007', '6.72', '22,127', '99.9', '220.1']
    overheads = ['+2.5%', '-2.8%', '-8.3%', '+0.8%', '+2.6%', '+2.6%',
                 '+41.0%', '-10.7%', '+57.3%', '+12.8%', '+23.2%', '+0.0%', '+2.9%']

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    colors_overhead = []
    for oh in overheads:
        val = float(oh.replace('%', '').replace('+', ''))
        if abs(val) < 1:
            colors_overhead.append('#e8f5e9')
        elif val > 5:
            colors_overhead.append('#ffcdd2')
        elif val > 0:
            colors_overhead.append('#fff9c4')
        elif val < -5:
            colors_overhead.append('#c8e6c9')
        else:
            colors_overhead.append('#e8f5e9')

    cell_text = list(zip(metrics, native_vals, container_vals, overheads))
    table_data = [[m, n, c, o] for m, n, c, o in cell_text]
    col_labels = ['Metric', 'Native', 'Container', 'Overhead']

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     cellLoc='center', loc='center', colWidths=[0.35, 0.2, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Style header
    for j in range(4):
        table[0, j].set_facecolor('#1565C0')
        table[0, j].set_text_props(color='white', fontweight='bold', fontsize=11)

    # Color overhead cells
    for i, color in enumerate(colors_overhead):
        table[i + 1, 3].set_facecolor(color)

    # Alternate row colors
    for i in range(len(metrics)):
        base_color = '#f5f5f5' if i % 2 == 0 else 'white'
        for j in range(3):
            table[i + 1, j].set_facecolor(base_color)

    ax.set_title('Overhead Summary: Native vs Container\n'
                 'ResNet-18 / CIFAR-10 / 2×H100 NVL / 10 Epochs / eBPF Profiling',
                 fontsize=14, fontweight='bold', pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "11_overhead_summary_table.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [11/11] Overhead Summary Table saved")


def main():
    print(f"Generating hardcoded plots to: {OUTPUT_DIR}")
    print("=" * 60)
    plot_training_loss()
    plot_training_accuracy()
    plot_throughput()
    plot_epoch_time()
    plot_total_time_comparison()
    plot_gpu_utilization()
    plot_syscall_comparison()
    plot_syscall_latency()
    plot_network_comparison()
    plot_sched_latency()
    plot_overhead_summary()
    print("=" * 60)
    print(f"All 11 plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
