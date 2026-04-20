import pandas as pd
import glob
import os

def clean_network_trace():
    print("--- Processing Network Trace ---")
    # Find the most recent network trace CSV
    network_files = glob.glob("network_trace_*.csv")
    if not network_files:
        print("No network trace found.")
        return
    
    latest_net_file = max(network_files, key=os.path.getctime)
    print(f"Loading: {latest_net_file}")
    
    # Load data
    df = pd.read_csv(latest_net_file)
    
    # 1. Sort chronologically by the kernel's microsecond timestamp
    df = df.sort_values(by="Timestamp(us)")
    
    # 2. Filter out the noise (Ignore tiny 8KB internal ACKs)
    # We only care about large physical network transfers (like the 256KB chunks or 44MB local hops)
    df_clean = df[df['Payload_Size_Bytes'] >= 250000].copy()
    
    if df_clean.empty:
        print("No large payload transfers found in this trace.")
        return

    # 3. Calculate Latency (Delta Time in milliseconds)
    # How long did it take between this chunk and the previous chunk?
    df_clean['Latency_Delta_ms'] = df_clean['Timestamp(us)'].diff() / 1000.0
    df_clean['Latency_Delta_ms'] = df_clean['Latency_Delta_ms'].fillna(0).round(2)
    
    # 4. Convert Bytes to Megabytes for easier reading
    df_clean['Payload_MB'] = (df_clean['Payload_Size_Bytes'] / (1024 * 1024)).round(2)
    
    # Rearrange columns for the final report
    cols = ['Time', 'Latency_Delta_ms', 'Direction', 'Src_IP', 'Dest_IP', 'Payload_MB']
    df_final = df_clean[cols]
    
    # Save the cleaned data
    output_name = "CLEANED_" + latest_net_file
    df_final.to_csv(output_name, index=False)
    print(f"Saved cleaned network data to: {output_name}")
    print(df_final.head(10).to_string(index=False))
    print("\n")


def clean_gpu_trace():
    print("--- Processing GPU Trace ---")
    # Find the most recent GPU trace CSV
    gpu_files = glob.glob("gpu_ioctl_trace_*.csv")
    if not gpu_files:
        print("No GPU trace found.")
        return
        
    latest_gpu_file = max(gpu_files, key=os.path.getctime)
    print(f"Loading: {latest_gpu_file}")
    
    # Load data
    df = pd.read_csv(latest_gpu_file)
    
    # 1. Aggregate: We don't need millions of rows, we just need to know how hard the GPU was working per second
    # Group by the human-readable 'Time' column and count the number of IOCTL commands
    df_agg = df.groupby('Time').size().reset_index(name='IOCTLs_per_sec')
    
    # 2. Categorize the GPU State
    # If it's processing > 50 commands a second, it's actively computing the PyTorch model. 
    # If it's < 50, it is idle/waiting on the network.
    df_agg['GPU_State'] = df_agg['IOCTLs_per_sec'].apply(lambda x: 'COMPUTING' if x > 50 else 'IDLE / WAITING')
    
    # Save the cleaned data
    output_name = "CLEANED_" + latest_gpu_file
    df_agg.to_csv(output_name, index=False)
    print(f"Saved cleaned GPU data to: {output_name}")
    print(df_agg.head(15).to_string(index=False))
    print("\n")

if __name__ == "__main__":
    # Ensure pandas is installed: pip install pandas
    clean_network_trace()
    clean_gpu_trace()