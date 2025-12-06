import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_all_neon_implementations():
    """
    Plots all NEON implementations for comparison across cores
    """
    files = [
        ("simd-math-arm-stats.csv", "Basic NEON Multiply", "blue"),
        ("neon-advanced-stats.csv", "Advanced NEON Multiply", "red"),
        ("neon-integer-stats.csv", "NEON Integer Operations", "green"),
    ]
    
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 8))

    has_data = False
    
    for csv_file, label, color in files:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df.columns = df.columns.str.strip()
            
            if "core_id" in df.columns and "throughput" in df.columns:
                ax.plot(df["core_id"], df["throughput"], marker="o", linestyle="-", 
                       label=label, color=color)
                has_data = True
            else:
                print(f"Warning: {csv_file} missing required columns")
        else:
            print(f"Warning: File {csv_file} not found")
    
    if has_data:
        ax.set_xlabel("Core ID")
        ax.set_ylabel("Throughput (ops/sec)")
        ax.set_title("NEON SIMD Implementations Comparison per Core")
        ax.grid(True)
        ax.set_xticks(range(0, 16))  # Assuming up to 16 cores
        ax.legend()
        
        output_path = "neon-implementations-comparison.png"
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    else:
        print("No valid data files found to plot")


def plot_neon_operations_comparison():
    """
    Creates separate plots for different NEON operation types
    """
    files = {
        "neon-add-stats.csv": "NEON Float Add Operations",
        "neon-mixed-stats.csv": "NEON Mixed Operations",
        "neon-double-stats.csv": "NEON Double Precision",
        "neon-fma-stats.csv": "NEON Fused Multiply-Add"
    }
    
    for csv_file, title in files.items():
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df.columns = df.columns.str.strip()
            
            if "core_id" in df.columns and "throughput" in df.columns:
                plt.style.use("seaborn-v0_8-whitegrid")
                fig, ax = plt.subplots(figsize=(14, 8))
                
                ax.plot(df["core_id"], df["throughput"], marker="o", linestyle="-", color="purple")
                ax.set_xlabel("Core ID")
                ax.set_ylabel("Throughput (ops/sec)")
                ax.set_title(title)
                ax.grid(True)
                ax.set_xticks(df["core_id"])
                
                output_path = f"{csv_file.replace('-stats.csv', '')}-throughput.png"
                plt.savefig(output_path, bbox_inches="tight")
                print(f"Plot saved to {output_path}")
                
                plt.close()  # Close the figure to free memory


if __name__ == "__main__":
    plot_all_neon_implementations()
    plot_neon_operations_comparison()