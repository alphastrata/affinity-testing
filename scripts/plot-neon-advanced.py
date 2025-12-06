import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

CSV_FILE = "neon-advanced-stats.csv"


def plot_throughput(csv_path):
    """
    Reads and plots the throughput data from the benchmark CSV.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Benchmark data file not found at '{csv_path}'")
        print(
            "Please run the benchmark first: ./target/release/neon-advanced --output neon-advanced-stats.csv"
        )
        return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if "core_id" not in df.columns or "throughput" not in df.columns:
        print(
            f"Error: CSV file '{csv_path}' is missing required 'core_id' or 'throughput' columns."
        )
        return

    unit = df["unit"].dropna().iloc[0] if "unit" in df.columns else "units"

    # Find the best performing core
    best_core_idx = df["throughput"].idxmax()
    best_core_id = df.loc[best_core_idx, "core_id"]
    best_throughput = df.loc[best_core_idx, "throughput"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), height_ratios=[3, 1])

    # Main plot: Throughput per core
    ax1.plot(df["core_id"], df["throughput"], marker="o", linestyle="-", linewidth=2, markersize=8, label=f'Performance ({unit})')

    # Highlight the best core
    ax1.scatter(best_core_id, best_throughput, color='red', s=150, zorder=5,
               label=f'Best Core {best_core_id}: {best_throughput:.2e} {unit}', edgecolors='black', linewidth=2)

    # Improve y-axis formatting and add unit explanation
    ax1.set_xlabel("Core ID")
    ax1.set_ylabel(f"Throughput ({unit})")

    # Format scientific notation more clearly
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    # Add unit explanation as text on the plot
    ax1.text(0.02, 0.98, f'1e9 = 1 billion, 1e6 = 1 million, etc.',
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax1.set_title("NEON Advanced Multiply Throughput per Core")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(df["core_id"])
    ax1.legend(loc='upper right')

    # Second plot: Aggregated core performance as horizontal bar chart
    df_sorted = df.sort_values('throughput', ascending=True)  # Sort for better visualization

    # Create a colormap based on performance
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_sorted)))

    ax2.barh(range(len(df_sorted)), df_sorted['throughput'], color=colors)
    ax2.set_xlabel(f"Throughput ({unit})")
    ax2.set_ylabel("Core ID")
    ax2.set_title("Aggregated Core Performance (Sorted)")
    ax2.set_yticks(range(len(df_sorted)))
    ax2.set_yticklabels([f"Core {cid}" for cid in df_sorted['core_id']])

    # Add values as text on bars
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        ax2.text(row['throughput'] * 0.5, i, f"{row['throughput']:.2e}",
                ha='center', va='center', fontweight='bold', color='white')

    plt.tight_layout()
    output_path = "neon-advanced-throughput.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    print(f"Best performing core: Core {best_core_id} with {best_throughput:.2e} {unit}")


if __name__ == "__main__":
    plot_throughput(CSV_FILE)
